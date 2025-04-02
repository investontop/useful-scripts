import phonenumbers
from phonenumbers import timezone, geocoder, carrier, is_valid_number

number = '+919644466735'

phone = phonenumbers.parse(number)
time = timezone.time_zones_for_number(phone)
carrier = carrier.name_for_number(phone, "en")
Country=geocoder.description_for_number(phone, "en")

formatted_number = phonenumbers.format_number(phone, phonenumbers.PhoneNumberFormat.INTERNATIONAL)

print('['+ Country + '] [' + formatted_number + '] [' + carrier + ']')

text = "Call me at 510-748-8230 if it's before 9:30, or on 703-4800500 after 10am."
for match in phonenumbers.PhoneNumberMatcher(text, "IN"):
    print(match)
    print(phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164))