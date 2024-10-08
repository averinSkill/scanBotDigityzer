import os
import zipfile
import telebot
import logging

from datetime import datetime
from pathlib import Path
from telebot import types

from config import TOKEN
from processing import digityzer


bot = telebot.TeleBot(TOKEN)

logging.basicConfig(level=logging.INFO, filename=datetime.now().strftime("%A_%d_%B_%Y_%I-%M%p") + ".log",filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")

path_to_result = Path(Path.cwd() / 'Result')
if not path_to_result.exists():
    path_to_result.mkdir(parents=False, exist_ok=False)


def save_project_name(message):
    chat_id = message.chat.id
    name = message.text
    # global project_name
    # project_name = name 
    keyboard = telebot.types.InlineKeyboardMarkup()
    button_save = telebot.types.InlineKeyboardButton(text="Сохранить",
                                                     callback_data='save_prj_name')
    button_change = telebot.types.InlineKeyboardButton(text="Отменить",
                                                       callback_data='cancel')
    keyboard.add(button_save, button_change)
    bot.send_message(chat_id, 'Сохранить название?')
    bot.send_message(chat_id, name, reply_markup=keyboard)


@bot.callback_query_handler(func=lambda call: call.data == 'save_prj_name')
def save_btn(call):
    message = call.message
    print("save_btn(call) message.text= ", message.text)
    # global project_name
    global prj
    prj = Project(message.text)
    logging.info(f"{call.from_user.id} start prj {message.text}.")
    msg_prj = prj.make_dir()
    logging.info(f"{call.from_user.id} make dir {msg_prj}.")
    print("save_btn= prj.path_to_project= ", prj.path_to_project)
    # print("save_btn= project_name= ", project_name)
    print("save_btn= prj= ", prj)  
    start_menu = types.ReplyKeyboardMarkup(True)
    start_menu.row('Закрыть проект')
    # bot.edit_message_text(chat_id=chat_id, message_id=message_id, 
    #                      text='Проект создан! Кидай фото', reply_markup=start_menu) 
    bot.send_message(message.chat.id, f'{msg_prj} Кидай фото', reply_markup=start_menu)
    

@bot.callback_query_handler(func=lambda call: call.data == 'cancel')
def cancel_btn(call):
    bot.send_message(call.from_user.id, "Жми /start.")
    

# @bot.callback_query_handler(func=lambda call: call.data == 'close_prj')
# def close_btn(call):
#     message = call.message
#     print(call.data)
#     # chat_id = message.chat.id
#     bot.register_next_step_handler(message, close_project)


@bot.message_handler(commands=['start'])
def menu(message):
    start_menu = types.ReplyKeyboardMarkup(True)
    start_menu.row('Создать проект')
    mesg = bot.send_message(message.chat.id, 'Привет, {0.first_name}! Добро пожаловать в бота дигитайзера оцифровки лекал Version 2.0'.format(message.from_user), reply_markup=start_menu)
    # bot.register_next_step_handler(mesg, test)


@bot.message_handler(func=lambda message: message.text == 'Создать проект')
def create_new_prj(message):
    chat_id = message.chat.id
    bot.send_message(chat_id, 'Введите название проекта')
    bot.register_next_step_handler(message, save_project_name)


@bot.message_handler(func=lambda message: message.text == 'Закрыть проект')
def close_prj(message):
    chat_id = message.chat.id
    start_menu = types.ReplyKeyboardMarkup(True)
    start_menu.row('Создать проект')
    bot.send_message(chat_id, 'Проект заархивирован. Получай!', reply_markup=start_menu)
    global prj
    prj.closed = True
    zipfile_name = prj.make_zip()
    print("close_prj(message)=", zipfile_name)
    bot.send_document(message.chat.id, open(zipfile_name, 'rb'))
    # bot.send_message(chat_id, f'{prj.photos}')
    del prj
    
    
@bot.message_handler(content_types=['document'])
def handle_file(message):
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    global prj
    if prj.path_to_project:
        save_path = prj.path_to_project / message.document.file_name
        prj.add_photo(message.document.file_name)
    else:
        save_path = path_to_result / message.document.file_name

    with save_path.open('wb') as new_file:
        new_file.write(downloaded_file)

    logging.info(f"{message.from_user.id} send photo {message.document.file_name}.")
    logging.info(f"photo {message.document.file_name} saved.")
    bot.reply_to(message, " Жди! Обрабатываю...")
    fln = digityzer(save_path, threshold=185)
    bot.send_document(message.chat.id, open(fln, 'rb'))
    


class Project:
    def __init__(self, name):
        self.name = name
        self.created = datetime.now()
        self.photos = list('')
        self.closed = False
        self.path_to_project = ''

    def display_data(self):  
        print(f'Название проекта: <<{self.name}>>. \nСоздан: {self.created}. \nСписок фоток {self.photos}')

    def make_dir(self):
        path_to_project = Path(Path.cwd() / self.name)
        if not path_to_project.exists():
            path_to_project.mkdir(parents=False, exist_ok=False)
            self.path_to_project = path_to_project
            return f'Проект <<{self.name}>> создан'
        else:
            return f'Проект <<{self.name}>> уже существует'

    def add_photo(self, photo_name):
        self.photos.append(photo_name)

    def make_zip(self):
        zipfile_name = self.path_to_project / (self.name + '.zip')
        print("zipfile_name= ", zipfile_name)
        with zipfile.ZipFile(zipfile_name, 'w', zipfile.ZIP_DEFLATED) as myzip:
             for file_n in os.listdir(self.path_to_project):
                  if file_n.endswith('.dxf'):
                #   if not file_n.startswith('.') and not file_n.endswith('.zip'):
                        print(file_n)
                        myzip.write(self.path_to_project / file_n, arcname=file_n)
        return zipfile_name

    def __del__(self):  
        class_name = self.__class__.__name__  
        print('{} уничтожен'.format(class_name))


if __name__ == '__main__':
    print('Бот запущен!')
    bot.infinity_polling()

