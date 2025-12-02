import httpx

url = 'https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m'
response = httpx.get(url)
data = response.json()

print('Current Weather in Berlin:')
print(f'  Temperature: {data["current"]["temperature_2m"]}°C')
print(f'  Wind Speed: {data["current"]["wind_speed_10m"]} km/h')
print(f'  Time: {data["current"]["time"]}')
print(f'\nHourly forecast has {len(data["hourly"]["time"])} entries')
