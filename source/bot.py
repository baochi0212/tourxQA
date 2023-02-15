import os
import telebot
import pandas as pd
from telegram.constants import ParseMode
from prettytable import PrettyTable
from underthesea import word_tokenize


BOT_TOKEN = os.environ['BOT_TOKEN']
bot = telebot.TeleBot(BOT_TOKEN)
working_dir = os.environ['dir']
google_dir = working_dir + '/crawler/googler'
source_dir = working_dir +'/source'

@bot.message_handler(commands=['hello'])
def send_welcome(message):
    bot.reply_to(message, "Hello I'm travelling chatbot!")

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Let's working man")
@bot.message_handler(commands=['command_travelling'])
def get_input(message):
    #send question
    text = "What is your command?"
    sent_msg = bot.send_message(message.chat.id, text)
    #message, callback, *args  
    bot.register_next_step_handler(sent_msg, main)

@bot.message_handler(commands=['ask_travelling'])
def get_input(message):
    #send question
    text = "What is your question?"
    sent_msg = bot.send_message(message.chat.id, text)
    #message, callback, *args  
    bot.register_next_step_handler(sent_msg, main_QA)
def main_IDSF(message):
    #command -> input file
    cmd = message.text
    os.system(f'touch ./input.txt && echo {cmd} > ./input.txt')
    #input and output txt file reset
    os.system("touch sample_output.txt")
    with open("input.txt", 'r') as f:
        f_write = open("sample_input.txt", "w")
        output_write = open("sample_output.txt", "w")
        for line in f.readlines():
            print(line)
            f_write.write(word_tokenize(line, format="text"))
    f_write.close()
    output_write.close()

    
    #idsf
    os.system('python predict.py --pretrained --model_type phobert --n_epochs 50 --train_batch_size 32 --eval_batch_size 32  --device cuda    --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.6 --learning_rate 5e-5 --predict_task "test example"')
    #answer
    bot.send_message(message.chat.id, open("./sample_output.txt", "r").readlines()[0].strip())
def main_QA(message):
    #get passage
    query = message.text
    #google search
    os.system(f'python {google_dir}/query.py --query "{query}"')
    #predict
    os.system('python predict.py --pretrained --model_type xlm-roberta-large --pretrained_model  xlm-roberta-large --device cuda --task QA --train_batch_size 16 --eval_batch_size 32 --tuning_metric F1_score --logging_step 1000 --n_epochs 5 --module_role QA')
    #read output
    for msg in open(source_dir + '/sample_output.txt', 'r'):
        if msg.strip() != 'NO ANSWER' and len(msg.strip()) > 0:
            bot.send_message(message.chat.id, msg.strip())

    # outputs = get_prediction(input)
    # table = PrettyTable(['text', 'label', 'probability'])
    # for (input, output, prob) in outputs:
    #     table.add_row([input.strip()[:20] + '...', output, prob])
    # bot.send_message(message.chat.id, "MY WARNING's MAYBE WRONG, BE CAUTIOUS!!!")
    # bot.reply_to(message, f'<pre>{table}</pre>', parse_mode=ParseMode.HTML)



bot.infinity_polling()