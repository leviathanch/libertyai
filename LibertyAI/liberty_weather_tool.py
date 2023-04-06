import requests
import datetime

def weather_getCmd(CityNameFormatted):
    GeocodingUrl = "https://geocoding-api.open-meteo.com/v1/search"
    ForecastUrl = "https://api.open-meteo.com/v1/forecast"
    #AirQualityUrl = "https://air-quality-api.open-meteo.com/v1/air-quality"
    cityinfoUrl = GeocodingUrl + "?name=" + CityNameFormatted + "&count=1"
    x = requests.get(cityinfoUrl)

    results = x.json()
    results = results['results'][0]
    url = ForecastUrl+"?timezone=auto&latitude="
    url += str(results['latitude'])
    url += "&longitude="
    url += str(results['longitude'])
    url += "&current_weather=true&hourly=relativehumidity_2m,apparent_temperature,surface_pressure,pressure_msl"
    x = requests.get(url)
    return x.json()

def translateweathercode(code):
    match code:
        case 0:
            return "Clear Sky"
        case 1:
            return "Mainly Clear"
        case 2:
            return "Partly Cloudy"
        case 3:
            return "Overcast"
        case 45:
            return "Fog"
        case 48:
            return "Depositing Rime Fog"
        case 51:
            return "Light Drizzle"
        case 53:
            return "Moderate Drizzle"
        case 55:
            return "Dense Drizzle"
        case 56:
            return "Light Freezing Drizzle"
        case 57:
            return "Dense Freezing Drizzle"
        case 61:
            return "Slight Rain"
        case 63:
            return "Moderate Rain"
        case 65:
            return "Heavy Rain"
        case 66:
            return "Light Freezing Rain"
        case 67:
            return "Heavy Freezing Rain"
        case 71:
            return "Slight Snow Fall"
        case 73:
            return "Moderate Snow Fall"
        case 75:
            return "Heavy Snow Fall"
        case 77:
            return "Snow Grains"
        case 80:
            return "Slight Rain Showers"
        case 81:
            return "Moderate Rain Showers"
        case 82:
            return "Violent Rain Showers"
        case 85:
            return "Slight Snow Showers"
        case 86:
            return "Heavy Snow Showers"
        case 95:
            return "Thunderstorm"
        case 96:
            return "Thunderstorm With Light Hail"
        case 99:
            return "Thunderstorm With Heavy Hail"

def get_current_weather(query):
    location = query.split('\n')[0].strip()
    now = datetime.datetime.now()
    reply = weather_getCmd(location);
    ret = f"""{translateweathercode(reply['current_weather']['weathercode'])}
Temperature: {reply['current_weather']['temperature']}°C
Feels like: {reply['hourly']['apparent_temperature'][now.hour]}°C
Wind speed: {reply['current_weather']['windspeed']}km/h
Wind direction {reply['current_weather']['winddirection']}°
Humidity {reply['hourly']['relativehumidity_2m'][now.hour]}%"""
    return ret
