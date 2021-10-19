#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='https://mainacademy.ua/'> <img src='https://mainacademy.ua/wp-content/uploads/2019/03/logo-main.png' alt = 'Хмм, щось з інтернетом'/></a>
# ___
# 
# # Module 2: Basic Python

# ## Lab work 2.2
# 
# 

# #### Мета: 
# 
# * навчитися працювати із основними інструкціями Python;
# * ознайомитися із базовими алгоритмами;

# ### Завдання 1

# Для трьох введених натуральних чисел (користувач вводить числа через пробіл) знайдіть мінімальне та максимальне

# In[22]:


a = list(map(int, input().split()))

print( "min =", min(a), "max =", max(a))


# ### Завдання 2

# Програма приймає на вхід два параметри: стать спортсмена(`sex = (f, m)`) та вік спортсмена(`age = [0,100]`).  Спортсмен-дівчина може брати участь у змаганні, якщо її вік більший 18, спортсмен-хлопець - якщо вік більший 16
# 
# Якщо введеніть не коректні дані, виведіть повідомлення про помилку

# In[26]:


sex = input()
age = float(input())

if sex = "f":
    
 print('Вітаємо на змаганнях')


# ### Завдання 3

# Для введеного натурального числа побудуйте драбинку. 
# 
# Наприклад, якщо N = 5, драбинка буде мати вигляд:
#     
# &
# 
# &&
# 
# &&&
# 
# &&&&
# 
# &&&&&

# In[1]:


n = int(input("Введіть число від 1 до 9 : "))
 
if n > 9 or n < 1 :
    print("Еrrоr")
else:
    s = ''
    for i in range(1, n+1):
        s += str(i)
        print(s)


# ### Завдання 4

# Із заданого речення виведіть кожне слово і його довжину

# In[1]:


sentence = 'My favourite tutor in academy is Ihor'

my_list = ["My", "favourite", "tutor", "in", "academy", "is", "Ihor"]

print(my_list[0:7])

len(my_list[0])


# ### Завдання 5

# Для заданого тексту реалізуйте:
# * Знайдіть кількість слів
# * Знайдіть кількість унікальних слів
# * Своірть словник, де ключе буде слово із тексту, значенням - частота зустрічання в тексті
# * Виведіть топ-3 слів, що найчастіше зустрічаються
# 
# Вважаємо, що слово це стрічка з трьох і більше букв, очищена від знаків пунктуації

# In[2]:


text = """До чого ж гарно і весело було в нашому горóді! Ото як вийти з сіней та подивись навколо — геть-чисто все зелене та буйне. А сад було як зацвіте весною! А що робилось на початку літа — огірки цвітуть, гарбузи цвітуть, картопля цвіте. Цвіте малина, смородина, тютюн, квасоля. А соняшника, а маку, буряків, лободи, укропу, моркви! Чого тільки не насадить наша невгамовна мати.

— Нічого в світі так я не люблю, як саджати що-небудь у землю, щоб проізростало. Коли вилізає з землі всяка рослиночка, ото мені радість,— любила проказувати вона.

Город до того переповнявсь рослинами, що десь серед літа вони вже не вміщалися в ньому. Вони лізли одна на одну, переплітались, душились, дерлися на хлів, на стріху, повзли на тин, а гарбузи звисали з тину прямо на вулицю.

А малини — красної, білої! А вишень, а груш солодких, було. як наїсися,— цілий день живіт як бубон.

І росло ще, пригадую, багато тютюну, в якому ми, маленькі, ходили, мов у лісі, в якому пізнали перші мозолі на дитячих руках.

А вздовж тину, за старою повіткою, росли великі кущі смородини, бузини і ще якихось невідомих рослин. Там неслися кури нишком од матері і різне дрібне птаство. Туди ми рідко лазили. Там було темно навіть удень, і ми боялись гадюки. Хто з нас у дитинстві не боявся гадюки, так за все життя й не побачивши її ніде?

Коло хати, що стояла в саду, цвіли квіти, а за хатою, проти сінешніх дверей, коло вишень,— поросла полином стара погребня з одкритою лядою, звідки завжди пахло цвіллю. Там, у льоху, в присмерку плигали жаби. Напевно, там водилися й гадюки.

На погребні любив спати дід.

У нас був дід дуже схожий на бога. Коли я молився богу, я завжди бачив на покуті портрет діда в старих срібнофольгових шатах, а сам дід лежав на печі і тихо кашляв, слухаючи своїх молитов.

У неділю перед богами горіла маленька синенька лампадка, в яку завжди набиралось повно мух. Образ святого Миколая також був схожий на діда, особливо коли дід часом підстригав собі бороду і випивав перед обідом чарку горілки з перцем, і мати не лаялась. Святий Федосій більш скидався на батька. Федосію я не молився, в нього була ще темна борода, а в руці гирлига, одягнена чомусь у білу хустку. А от бог, схожий на діда, той тримав в одній руці круглу сільничку, а трьома пýчками другої неначе збирався взяти зубок часнику
Звали нашого діда, як я вже потім довідавсь, Семеном. Він був високий і худий, і чоло в нього високе, хвилясте довге волосся сиве, а борода біла. І була в нього велика грижа ще з молодих чумацьких літ. Пахнув дід теплою землею і трохи млином. Він був письменний по-церковному і в неділю любив урочисто читати псалтир. Ні дід, ні ми не розуміли прочитаного, і це завжди хвилювало нас, як дивна таємниця, що надавала прочитаному особливого, небуденного смислу.

Мати ненавиділа діда і вважала його за чорнокнижника. Ми не вірили матері і захищали діда од її нападів, бо псалтир всередині був не чорний, а білий, а товста шкіряна палітурка — коричнева, як гречаний мед чи стара халява. Зрештою, мати крадькома таки знищила псалтир. Вона спалила його в печі по одному листочку, боячись палити зразу весь, щоб він часом не вибухнув і не розніс печі.

Любив дід гарну бесіду й добре слово. Часом по дорозі на луг, коли хто питав у нього дорогу на Борзну чи на Батурин, він довго стояв посеред шляху і, махаючи пужалном, гукав услід подорожньому:

— Прямо, та й прямо, та й прямо, та й нікуди ж не звертайте!.. Добра людина поїхала, дай їй бог здоров'я,— зітхав він лагідно, коли подорожній нарешті зникав у кущах.

— А хто вона, діду, людина ота? Звідки вона?

— А бог її знає, хіба я знаю... Ну, чого стоїш як укопаний? — звертався дід до коня, сідаючи на воза.— Но, трогай-бо, ну...

Він був наш добрий дух лугу і риби. Гриби й ягоди збирав він у лісі краще за нас усіх і розмовляв з кіньми, з телятами, з травами, з старою грушею і дубом — з усім живим, що росло і рухалось навколо.
Більш за все на світі любив дід сонце. Він прожив під сонцем коло ста літ, ніколи не ховаючись у холодок. Так під сонцем на погребні, коло яблуні, він і помер, коли прийшов його час.

Дід любив кашляти. Кашляв він часом так довго й гучно, що скільки ми не старалися, ніхто не міг його як слід передражнити. Його кашель чув увесь куток. Старі люди по дідовому кашлю вгадували навіть погоду.
"""
len(text)
unique_text = list(set(text))


# ### Завдання 6 

# Напишіть функції, що:
# * рохраховує мінімальне та максимальне значення
# * середнє значення послідовності
# * медіану послідовності
# * частоту кожної цифри в послідовності
# * середнє квадратичне відхилення
# 
# Послідовність чисел ганеруємо рандомно

# In[23]:


import random
import statistics

#генеруэмо тисячу чисел в діапазоні від 0 до 10
list_with_random_numbers = [random.randint(0,100) for i in range(1000)]
print("Мінімальне", min(list_with_random_numbers))

print("Максимальне", max(list_with_random_numbers))

average = sum(list_with_random_numbers) / len(list_with_random_numbers)

print('Середнє значення:' ,average)


print('Медіана:', statistics.median(list_with_random_numbers))


# In[ ]:





# In[ ]:





# In[ ]:



