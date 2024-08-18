# input are: Gibb,Robin:22/December/1949
# output are: Robin Gibb: December 22 1949

str_data = input("Enter A data: ").strip()

str_data = 'Gibb,Robin:22/December/1949'

name, date_str = str_data.split(':')
last_name, first_name = name.split(',')
day, month, year = date_str.split('/')

out = f"{first_name} {last_name}: {month} {day} {year}"
print(out)
