# Streaming TTS + Offline STT Pipeline
## Архитектура

- **tts-service**: Синтез речи со стриминговой выдачей PCM
- **asr-service**: Распознавание речи по файлу (английский)
- **gateway**: Единая точка входа для клиента

## Быстрый старт

1. Клонируйте репозиторий
2. Загрузите requiments
3. Создайте `.env` файл из `.env.example`
4. Запустите сервисы:

```bash
docker-compose up -d
