# Проверка evaluator и A/B — 25 октября 2026

Добавлены `evaluate run`, `evaluate score`, архивирование всех случаев,
ручные смысловые решения с проверкой происхождения и две версии prompt.
Реальные A/B-записи хранятся отдельно:
[A](../ab-A-2026-10-25/manifest.json), [B](../ab-B-2026-10-25/manifest.json).
Полный разбор: [docs/ab-report.md](../../docs/ab-report.md).

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m cartoon_rag.cli validate-data
QDRANT_INTEGRATION_URL=http://127.0.0.1:6333 RUN_SEMANTIC_SMOKE=1 \
  PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
git diff --check
```

- [tests-local.log](tests-local.log): 120 прошли, 3 интеграционных пропущены.
- [tests-live.log](tests-live.log): все 123 теста прошли, пропусков нет,
  с реальными Qdrant/E5. Gemini в unit-тестах — транспортная заглушка.
- [validate-data.json](validate-data.json): 12 активных карточек, 10 случаев.

15 новых тестов относительно этапа HTTP-исправления проверяют сохранение
раздельных артефактов, Recall по родительским источникам, неполные решения,
привязку к trace и rubric, контроль статуса/фильтров, ошибки и знаменатели,
остановку на 401/403/429, защиту от перезаписи и небезопасных ID, CLI и версии prompt.
При повторном скоринге частичная разметка не оставляет старое accepted-решение.

В настоящих A/B-прогонах отдельно выполнены все 10 случаев для каждой версии,
по 7 запросов Gemini (14 всего), без ошибок. Приёмка Codex: A — 9/10,
B — 10/10. Recall@3 — 100%, фильтры — 100% в обоих; контексты/ID/планы/
вычисления и runtime-параметры совпали, кроме версии prompt. Это настоящие
записи приложения; unit-тесты не используются как замена A/B-эксперимента.

Проверено на Python 3.13.15, macOS arm64. Каталог, вопросы и зависимости не
изменены. Ключ остаётся в `.env`; настройки целиком не сериализуются.
