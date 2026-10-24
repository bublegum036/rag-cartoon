# Обрыв HTTP-ответа Gemini — 25 октября 2026

Gemini HTTP-адаптер теперь перехватывает `http.client.HTTPException`, включая
`IncompleteRead`, и преобразует их в `GenerationUnavailable`. Частичный JSON
не становится ответом `insufficient`; повторов и переключений модели нет.

Два новых теста воспроизводят обрыв настоящим `HTTPResponse`: заголовок
обещает 100 байт, `BytesIO` содержит только 14 байт тела. Сеть не используется.
Проверены адаптер и полный путь через `SearchService`/генератор до CLI.
До исправления тесты падали с `IncompleteRead`; после него CLI возвращает
JSON `GenerationUnavailable`, код 1, пустой stdout и не создаёт успешную трассу.
Также проверены закрытие ответа, отсутствие ключа/traceback в выводе и один вызов.

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_gemini.py -v
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m cartoon_rag.cli validate-data
git diff --check
```

- [tests-before.log](tests-before.log): 13 тестов Gemini, два новых завершились ошибкой до исправления.
- [tests-local.log](tests-local.log): всего 108 тестов, 105 прошли, 3 интеграционных пропущены.
- [validate-data.json](validate-data.json): 12 активных карточек, 10 вопросов.

Проверено на Python 3.13.15. Настоящие Gemini, Qdrant и E5 повторно не вызывались.
Предыдущие сквозные журналы описывают состояние до этой правки. Каталог,
вопросы, зависимости, prompt и формат REST-запроса не менялись.
