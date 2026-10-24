# Чистая установка из lock — 25 октября 2026

Проверен `requirements/search-macos-arm64-py313.lock` на CPython 3.13.15,
macOS arm64, pip 26.2.1. Это проверка воспроизводимости Python-окружения;
полный RAG и A/B не выполнялись.

## Что выполнено

Создана новая `.venv` в `/private/tmp/cartoon-rag-lock-xe3e_thp/clean/`
без системных пакетов. Основная `.venv` проекта не переустанавливалась.
Все команды проверки выполнены из корня проекта, `PYTHONPATH` удалён
из окружения процесса. Использовались общие кэши загрузок/модели
и существующий локальный Qdrant 1.17.0.

1. `pip install --require-hashes --only-binary=:all: -r requirements/search-macos-arm64-py313.lock`.
2. `pip install --no-deps --no-build-isolation -e ".[search]"`.
3. `pip check` и сверка установленных версий с lock и предыдущим окружением.
4. Все тесты с `QDRANT_INTEGRATION_URL=http://127.0.0.1:6333` и `RUN_SEMANTIC_SMOKE=1`.
5. Установленный CLI: `cartoon-rag validate-data` и `cartoon-rag search "море и дружба" --duration-lt 90`.

Все команды завершились с кодом 0. Все 29 зафиксированных версий совпали;
28 прежних поисковых пакетов не обновлялись, добавлен setuptools 84.2.1.
Все 41 тест прошли без пропусков, включая настоящие Qdrant и E5.
Валидация: 12 активных карточек, 10 вопросов. Поиск включил film-01,
все найденные карточки удовлетворяют `< 90`.

## Артефакты

- [summary.json](summary.json): SHA-256 lock, число пакетов и результаты сверок.
- [environment.json](environment.json): Python, платформа, изоляция среды и инструменты генерации.
- [commands.json](commands.json): фактические команды, параметры тестов и коды завершения.
- [install-locked.log](install-locked.log), [install-project.log](install-project.log): журналы установки.
- [pip-check.log](pip-check.log), [installed.log](installed.log): проверка совместимости и версии.
- [tests.log](tests.log): все 41 тест, без пропусков.
- [validate-data.log](validate-data.log): результат проверки каталога и вопросов.
- [search.log](search.log), [search.json](search.json): исходный вывод поиска и отдельно JSON.

Генерация выполнена pip-tools 7.6.2: разрешение версий и сборочных зависимостей,
загрузка совместимых wheels, затем расчёт хешей локально без индекса.
Первоначальная генерация хешей для дистрибутивов других платформ остановлена
до установки. Её промежуточные загрузки и журналы находятся в игнорируемой
`.cache/`. В итоговом lock нет ссылок на локальную wheelhouse.
Команды установки и генерации: [requirements/README.md](../../requirements/README.md).

Другие версии Python и платформы не проверены. Установка не была проверкой
с нуля всей инфраструктуры: Qdrant и кэш весов уже существовали. Тесты базы
использовали собственные случайные alias; основной индекс не переиндексировался.
