# Зафиксированные зависимости поиска

`search-macos-arm64-py313.lock` предназначен для CPython 3.13 на macOS arm64.
Он включает 29 пакетов: зависимости extra `search` и backend сборки проекта,
с точными версиями и SHA-256-хешами выбранных wheels. Пакет `cartoon-rag`
устанавливается из текущего checkout отдельным шагом.

## Установка

Из корня проекта в новой среде:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes --only-binary=:all: -r requirements/search-macos-arm64-py313.lock
python -m pip install --no-deps --no-build-isolation -e ".[search]"
python -m pip check
```

Первый шаг установки проверяет хеши всех внешних пакетов и разрешает только
готовые wheels. Второй использует уже установленные зависимости и backend
сборки: `--no-deps` отключает разрешение зависимостей, `--no-build-isolation`
предотвращает загрузку отдельного backend вне lock. Локальный исходный код
идентифицируется Git-коммитом, веса E5 — отдельной ревизией в настройках.
pip поставляется с `venv`; при проверке использован pip 26.2.1.

## Перегенерация

Инструмент генерации устанавливать в отдельную временную `.venv` на той же
версии Python и платформе, чтобы не добавлять его в окружение приложения:

```bash
python -m pip install pip-tools==7.6.2
python -m piptools compile --extra search --all-build-deps --allow-unsafe --strip-extras --output-file requirements/search-macos-arm64-py313.lock pyproject.toml
python -m pip download --only-binary=:all: -r requirements/search-macos-arm64-py313.lock --dest .cache/lock-wheels
python -m piptools compile --extra search --all-build-deps --allow-unsafe --strip-extras --generate-hashes --no-build-isolation --no-index --find-links .cache/lock-wheels --no-emit-options --output-file requirements/search-macos-arm64-py313.lock pyproject.toml
```

Все команды выполняются из корня проекта. Проверенное окружение генератора:
Python 3.13.15, pip 26.2.1, pip-tools 7.6.2 и setuptools 84.2.1.
Первый проход разрешает версии, второй скачивает совместимые wheels,
третий рассчитывает хеши локально, без обращения к индексу пакетов.
`--no-emit-options` исключает локальный путь wheelhouse из итогового lock.
После первого прохода файл временно не содержит хешей: для установки
использовать только результат всех трёх команд с `--require-hashes`.

Источник прямых зависимостей —
`pyproject.toml`. Первоначальные версии поисковых пакетов взяты из проверенного
окружения `artifacts/verification-2026-10-25/dependencies.txt`;
генератор повторно разрешает зависимости и сохраняет совместимые версии
существующего lock. Для намеренного обновления добавить `--upgrade`
или `--upgrade-package <package>` к первому проходу, затем повторить остальные
команды, чистую установку и проверки.
Хеши и принципы генерации описаны в
[документации pip-tools](https://pip-tools.pypa.io/en/stable/).

Не считать этот файл проверенным lock для других платформ и версий Python.
Набор зависимостей и выбранные wheels зависят от окружения.
Для новой цели нужен отдельный lock и прогон.
Исторический `dependencies.txt` сохранён как запись прежнего окружения.

25 октября 2026 года установка в новую временную `.venv` прошла с проверкой
хешей; все 41 тест прошли без пропусков с Qdrant/E5. Проверены CLI,
совпадение всех версий с lock и сохранение прежних поисковых версий.
Журнал: [чистая установка](../artifacts/lock-verification-2026-10-25/README.md).
