import phonenumbers
from phonenumbers import timezone, geocoder, carrier

number = '+919677168734'

phone = phonenumbers.parse(number)
time = timezone.time_zones_for_number(phone)
carrier = carrier.name_for_number(phone, "en")
reg=geocoder.description_for_number(phone, "en")
print(reg)  