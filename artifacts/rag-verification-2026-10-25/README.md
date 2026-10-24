# Gemini и сквозной `ask` — 25 октября 2026

Реализованы Gemini REST-генератор, строгий JSON-декодер, сверка снимков
JSON/Qdrant и команда `ask`. Python 3.13.15, macOS arm64, Qdrant 1.17.0,
E5 с прежней закреплённой ревизией, Gemini `gemini-3.1-flash-lite`,
prompt `grounded-ru-v1`. Python-зависимости не менялись.

Пользователь выбрал бесплатный вариант Gemini и добавил ключ локально в `.env`.
Приложение не определяет billing-тариф ключа; проект должен оставаться
Free Tier без подключённой оплаты. Автоматических повторов и смены модели нет.
Ключ не включён в сохранённые файлы. Во время этого этапа выполнены два
успешных запроса генерации Gemini; unit-тесты не вызывают этот API.

## Команды

Из корня проекта:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m cartoon_rag.cli validate-data
QDRANT_INTEGRATION_URL=http://127.0.0.1:6333 RUN_SEMANTIC_SMOKE=1 \
  PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v

PYTHONPATH=src .venv/bin/python -m cartoon_rag.cli ask "Хочу мультфильм про море и дружбу, короче 90 минут." --trace artifacts/rag-verification-2026-10-25/sea.trace.json
PYTHONPATH=src .venv/bin/python -m cartoon_rag.cli ask "Подойдёт ли «Коралловая почта», если у меня только 90 минут?" --trace artifacts/rag-verification-2026-10-25/compare.trace.json
PYTHONPATH=src .venv/bin/python -m cartoon_rag.cli ask "Сколько он длится?" --trace artifacts/rag-verification-2026-10-25/clarify.trace.json
PYTHONPATH=src .venv/bin/python -m cartoon_rag.cli ask "Кто озвучивает девочку в «Маяке для кита»?" --trace artifacts/rag-verification-2026-10-25/unknown.trace.json
git diff --check
```

Для сети Gemini, Hugging Face и Docker/Qdrant использован доступ за пределами
песочницы. Основной индекс читался, без повторной публикации; интеграционные
тесты создавали и удаляли только свои случайные alias/коллекции.

## Результаты

- [tests-local.log](tests-local.log): 103 прошли, 3 интеграционных пропущены.
- [tests-live.log](tests-live.log): все 106 тестов прошли без пропусков,
  с настоящими Qdrant/E5. Gemini в этих тестах — транспортная заглушка.
- 18 новых тестов покрывают JSON-схему/типы, источники, ошибки квоты,
  отсутствие повторов, скрытие ключа, fingerprint, фиксацию коллекции,
  проверку metadata, сборку приложения и ранние ответы CLI.
- Настоящий тест Qdrant дополнен проверкой: после публикации версии 2
  закреплённый старый поиск видит 78 минут; связать старый JSON с новым
  alias нельзя, а новый совпадающий снимок видит 92 минуты.
- [validate-data.json](validate-data.json): 12 активных карточек и 10 вопросов.
- [manifest.json](manifest.json): модель, prompt, параметры и хеши данных/кода,
  без настроек и ключей API.

### Два настоящих ответа Gemini

- [sea.answer.json](sea.answer.json), [sea.trace.json](sea.trace.json):
  `answered`, «Маяк для кита», 78 минут, объяснение дружбы и источник
  `film-01#plot`. Проверены статус, соответствие морскому сюжету и дружбе
  по самой карточке, длительность и ссылки. Все девять retrieved-карточек
  удовлетворяют `< 90`, в рекомендации один фильм. Первым в выдаче был
  film-11; правильная карточка film-01 находилась в выбранном контексте.
- [compare.answer.json](compare.answer.json), [compare.trace.json](compare.trace.json):
  `answered`, отказ по бюджету: 96 минут, превышение на 6 относительно 90,
  источник `film-02#plot`. Трасса содержит вычисленное кодом значение +6;
  карточка не удалена фильтром. Итог просмотрен по фактам, сравнению и ссылкам.

`sea.stderr.log` и `compare.stderr.log` содержат только служебный прогресс
загрузчика E5; ошибки API не возникли. Модель и источники автоматически
не переключались, платные инструменты поиска не подключались.

### Ранние ответы без Gemini

- [clarify.answer.json](clarify.answer.json), [clarify.trace.json](clarify.trace.json):
  `clarify`, запрос названия, пустые retrieved/context.
- [unknown.answer.json](unknown.answer.json), [unknown.trace.json](unknown.trace.json):
  `insufficient`, отсутствие сведений об озвучивании, пустые источники.

## Границы проверки

Это несколько сквозных проверок, не A/B на всех 10 вопросах. Recall@k и доля
принятых ответов для всего набора не вычислялись. Просмотр двух ответов по
фактам не заменяет отдельный evaluator и проверку остальных случаев.
JSON-схема и существующая ссылка не доказывают смысловую истинность ответа.
Следующий этап — `evaluate`, затем наблюдаемая ошибка A, одна правка и B.

Контракты сверены с первичными источниками:
[структурированный JSON](https://ai.google.dev/gemini-api/docs/generate-content/structured-output),
[REST API](https://ai.google.dev/api/generate-content),
[модель](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite),
[тарифы](https://ai.google.dev/gemini-api/docs/pricing).
