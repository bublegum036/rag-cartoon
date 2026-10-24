# Проверка Qdrant и E5 — 25 октября 2026

Это проверка провайдеров и CLI поиска. Полный RAG и A/B здесь не выполнялись.

Окружение: Python 3.13.15, macOS arm64, Qdrant 1.17.0,
FastEmbed 0.9.0, ONNX Runtime 1.31.0. Веса `intfloat/multilingual-e5-small`,
ревизия `fd1525a9fd15316a2d503bf26ab031a61d056e98`, 384 измерения.

## Команды и результаты

Команды выполнялись из корня проекта. `.env` уже существовал; секреты не
копировались в артефакты. Для сетевых операций и Docker потребовался доступ
за пределами песочницы; первоначальные попытки внутри неё завершились
ошибками разрешения DNS или `Operation not permitted`.

```bash
.venv/bin/python -m pip install -e ".[search]"
.venv/bin/python -m pip check
docker compose up -d qdrant

QDRANT_INTEGRATION_URL=http://127.0.0.1:6333 RUN_SEMANTIC_SMOKE=1 \
  PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v

PYTHONPATH=src .venv/bin/python -m cartoon_rag.cli validate-data
PYTHONPATH=src .venv/bin/python -m cartoon_rag.cli index
PYTHONPATH=src .venv/bin/python -m cartoon_rag.cli search "море и дружба" --duration-lt 90
PYTHONPATH=src .venv/bin/python -m cartoon_rag.cli search "девочка и кит чинят маяк" --limit 3
```

- Установка и `pip check` завершились успешно.
- Все 41 тест прошли, пропусков нет; полный журнал — [tests.log](tests.log).
- Проверены фильтры до top-k, границы `<`/`<=`, повторная публикация,
  переход film-01 с 78 на 92 минуты, сбой загрузки и пустой снимок.
- Семантический тест подтвердил film-01 и film-05 в top-3 для кита и робота;
  запрос про море с `< 90` включил film-01 и соблюдал фильтр для всех hits.
- [validate-data.json](validate-data.json): 12 активных карточек, 10 вопросов.
- [index.json](index.json): основной alias `cartoons`, 12 фильмов, 12 частей,
  идентификатор модели и fingerprints снимка.
- [search-sea.json](search-sea.json): film-01, film-07, film-11, film-09,
  film-03; все короче 90 минут. Порядок — по оценке близости, смысловая
  приёмка всех кандидатов как рекомендаций не выполнялась.
- [search-whale.json](search-whale.json): film-01, film-11, film-08;
  film-01 на первом месте. JSON содержит части и сгруппированные карточки.
- [environment.json](environment.json): фактическая версия сервера и состояние
  после проверок. Тестовые alias/коллекции удалены; остался основной индекс.
- [dependencies.txt](dependencies.txt): снимок установленных пакетов,
  без editable-ссылки на локальный проект. Это не lock с хешами;
  чистая установка по этому списку не проверена.

Сервис Qdrant оставлен запущенным. Каталог и 10 evaluation cases не изменялись.
Финальные ответы, Recall@k и answer acceptance rate не вычислялись:
планировщик, точные операции и генератор пока не реализованы.
