'''
前置: 1. ngrok 2. Linebot developer 
1. 至ngrok官網下載病解壓縮執行檔，依據官網給定token執行 
    >> ngrok config add-authtoken "token"
    >> ngrok http 5000
    (flask預設port 5000)

2. 至linebot官網開啟新的聊天身分 (Message API)
    複製Channel access token並取代本py檔的access_token
    複製Channel secret 並取代本py檔的secret
    其他設置:
        關閉聊天功能
        打開Webhook
        關閉自動回應訊息
    將ngrok http 5000運行畫面的Forwarding網址到ngrok-free.app複製到Webhook URL

3. 執行此程式 
    >> python linebot_demo.py 

4. 在linebot官網驗證Webhook，點擊verify
    若不成功，檢查是否ngrok及flask(本py檔)皆以運行，並且使用相同port。

參考來源: https://hackmd.io/@Lisa304/HJCVkmE8h
        https://www.learncodewithmike.com/2020/06/python-line-bot.html
'''

from flask import Flask, request

# 載入 json 標準函式庫，處理回傳的資料格式
import json
import random
# 載入 LINE Message API 相關函式庫
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, AudioMessage
from linebot.models import StickerSendMessage   # 載入 StickerSendMessage 模組
import speech_recognition as sr
import os, io,time, random
from pprint import pformat
from opencc import OpenCC
from itertools import chain
from datetime import datetime
# from pydub import AudioSegment
from transformers import OpenAIGPTLMHeadModel, BertTokenizer
import torch
import torch.nn.functional as F
import soundfile
from langconv import Converter # 簡繁體轉換
from espnet2.bin.asr_inference import Speech2Text
from pydub import AudioSegment

def tokenize(obj):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)

cc = OpenCC('tw2s')
SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]", "[speaker1]", "[speaker2]"]
device = torch.device("cuda")
stickerlist = [[2014,446],[10877,789],[11087934,6362],[11088035,6370],[11825376,6632],[11825382,6632],[11825393,6632],[11825394,6632],[16581290,8525],[16581301,8525],[52002736,11537],[52002753,11537],[52002768,11537],[52002758,11537],[51626532,11538],[51626511,11538],[52114110,11539]]
random.seed(42)
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)

tokenizer_class = BertTokenizer
# model_class = OpenAIGPTLMHeadModel if not args.gpt2 else GPT2LMHeadModel
model_class = OpenAIGPTLMHeadModel
tokenizer = tokenizer_class.from_pretrained("Dialog_model/0507/", do_lower_case=True)
model = model_class.from_pretrained("Dialog_model/0507/")
model.to(device)
model.eval()

speech2text = Speech2Text('config.yaml','40epoch.pth') # ASR 模型

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):

    assert logits.dim() == 1  
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)


        sorted_indices_to_remove = cumulative_probabilities > top_p

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0


        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def build_input_from_segments(history, reply, tokenizer, with_eos=True):

    bos, eos, pad, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    sequence = [[bos]] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:])
                                          for _ in s]
    return instance, sequence


def sample_sequence(history, tokenizer, model, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(70):
        instance, sequence = build_input_from_segments(history, current_output, tokenizer, with_eos=False)
        input_ids = torch.tensor(instance["input_ids"], dtype=torch.long, device='cuda').unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], dtype=torch.long, device='cuda').unsqueeze(0)

        logits, *_ = model(input_ids, token_type_ids=token_type_ids)
        logits = logits[0, -1, :] / 0.7
        logits = top_filtering(logits, top_k=0, top_p=0.9)
        probs = F.softmax(logits, dim=-1)

        # prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        prev = torch.topk(probs, 1)[1] 
        if i < 1 and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def linebot():
    body = request.get_data(as_text=True)                    # 取得收到的訊息內容
    try:
        json_data = json.loads(body)                         # json 格式化訊息內容
        access_token = 'gWzevA8DChbTrfIpQ0R8HIe068Bjuhqe0A2LqVz++9ktbUdn+fb4wfj8qF9jPuuCfjJ0ZAUL0jQ8VxTv8jolfuD7LK9w816JILne1JFbvkPTsCk6NlI9JUyhRBa+COCTVmteKdBUol2g2WOg+BISbwdB04t89/1O/w1cDnyilFU='
        secret = '77d2b37d896e4731303b70ef331bc2e3'
        line_bot_api = LineBotApi(access_token)              # 確認 token 是否正確
        handler = WebhookHandler(secret)                     # 確認 secret 是否正確
        signature = request.headers['X-Line-Signature']      # 加入回傳的 headers
        handler.handle(body, signature)                      # 綁定訊息回傳的相關資訊
        tk = json_data['events'][0]['replyToken']            # 取得回傳訊息的 Token
        receive_type = json_data['events'][0]['message']['type']     # 取得 LINE 收到的訊息類型
        if receive_type == 'text':
            msg = json_data['events'][0]['message']['text']  # 取得 LINE 收到的文字訊息
            print(f'User: {msg}')                               
            history =[]
            raw_text = " ".join(list(cc.convert(msg).replace(" ", "")))
            history.append(tokenize(raw_text))
            with torch.no_grad():
                out_ids = sample_sequence(history, tokenizer, model)
            history.append(out_ids)
            history = history[-(2 * 5 + 1):]
            out_text = Converter('zh-hant').convert(tokenizer.decode(out_ids, skip_special_tokens=True).replace(' ','')).replace('幺','麼')
            reply = out_text                                      # Change your reply into your predict result of chatbot model
            line_bot_api.reply_message(tk,TextSendMessage(reply))# 回傳訊息
        elif receive_type == 'audio':
            # print(json_data)
            UserID = json_data['events'][0]['source']['userId']
            temp_path = 'linebot_record/' + UserID + '.mp3'
            audio_content = line_bot_api.get_message_content(json_data['events'][0]['message']['id'])
            with open(temp_path, 'wb') as fd:
                for chunk in audio_content.iter_content():
                    fd.write(chunk)        
            fd.close()
            #轉檔
            dst=temp_path.replace("mp3","wav")
            sound = AudioSegment.from_file(temp_path)
            sound.export(dst, format="wav")
            
            y, sr = soundfile.read(dst)
            msg = speech2text(y)[0][0]
            print(msg)
            # stickerId = 13 # 取得 stickerId
            # packageId = 1 # 取得 packageId
            # sticker_message = StickerSendMessage(sticker_id=stickerId, package_id=packageId) # 設定要回傳的表情貼圖
            # line_bot_api.reply_message(tk,sticker_message)

            history =[]
            raw_text = " ".join(list(cc.convert(msg).replace(" ", "")))
            history.append(tokenize(raw_text))
            with torch.no_grad():
                out_ids = sample_sequence(history, tokenizer, model)
            history.append(out_ids)
            history = history[-(2 * 5 + 1):]
            out_text = Converter('zh-hant').convert(tokenizer.decode(out_ids, skip_special_tokens=True).replace(' ','')).replace('幺','麼')
            reply = out_text                                      # Change your reply into your predict result of chatbot model
            line_bot_api.reply_message(tk,TextSendMessage(reply))# 回傳訊息
        else :
            # print(json_data['events'][0])
            # print(json_data)
            sticker = random.choice(stickerlist)
            stickerId = sticker[0] # 取得 stickerId
            packageId = sticker[1] # 取得 packageId
            sticker_message = StickerSendMessage(sticker_id=stickerId, package_id=packageId) # 設定要回傳的表情貼圖
            line_bot_api.reply_message(tk,sticker_message)# 回傳訊息
    except:
        print('body')                                          # 如果發生錯誤，印出收到的內容
    return 'OK'                                              # 驗證 Webhook 使用，不能省略


    

if __name__ == "__main__":
    app.run()