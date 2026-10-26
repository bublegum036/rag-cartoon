# Первый эксперимент с проверяемыми цитатами

26 октября 2026 выполнены те же четыре вопроса. Изменён режим генератора:
модель выбирает только существующие ID, код копирует цитаты и метаданные,
точные операции выполняет без LLM. Это сравнение компонента генерации,
не только prompt. Базовый коммит перед изменениями — 67c77ae.

Mode=extractive, prompt=extractive-ru-v1, renderer=extractive-renderer-v1.
Модель gemini-3.1-flash-lite, лимит 2048, retrieval_k=10, context_movies=3,
пауза 10 секунд. Общие runtime-поля совпали с v2, кроме prompt; вопросы,
rubric, подготовки и физический snapshot идентичны. Формат артефактов 2.

| Показатель | Результат |
| --- | --- |
| Случаев / подготовок | 4 / 4 |
| Ответов / технических ошибок / не выполнено | 2 / 2 / 0 |
| Принято Codex, общий знаменатель | 2/4 — 50% |
| Recall@1/@3, три случая с эталоном | 100% |
| Фильтр длительности | 1/1 |
| Запросов Gemini / успешных выборов | 3 / 1 |
| Ответов без Gemini | 1 |

| Случай | Итог |
| --- | --- |
| sea-short | HTTP 503, ответа нет, подготовка сохранена |
| space | HTTP 503, ответа нет, подготовка сохранена |
| mechanical-whale | Принят: точная цитата про рычаг/лебёдку и проход лодок |
| count-all | Принят: 40 и все 40 источников, без вызова Gemini |

run завершился с кодом 1, manifest.state=completed_with_errors. HTTP 503
не повторялись и не превращались в insufficient; модель и billing не менялись.
Два сохранённых ответа приняты, но ошибки остаются в знаменателе.
Это не доказательство ухудшения/улучшения смысла относительно v2:
двух ответов нет из-за API. Полный набор 20 не запускался; default — grounded/v2.

## Проверка

Codex прочитал text/reason по прежним rubric, фактам, ограничениям и ссылкам.
Оценка не независимая и не слепая. У mechanical-whale причина и применение
результата есть в дословной цитате, title/year/duration взяты из карточки.
У count-all проверены ExactResult=40 и все 40 источников. Reviews связаны
с case/trace хешами, критерии не менялись.

У sea-short реального ответа нет. [offline-rendering-example.json](offline-rendering-example.json)
отдельно показывает сборку из заранее известного ID: 78 минут и исходная
цитата без «в благодарность». Это офлайн-регрессионный пример, не ответ
модели и не дополнительный принятый случай. Тесты также отклоняют попытки
прислать выдуманную цитату/ID или свободный текст.

## Артефакты

- [manifest.json](manifest.json), [command.txt](command.txt), [prompt.txt](prompt.txt).
- [results.json](results.json), [metrics.json](metrics.json), [reviews.json](reviews.json).
- [verification.json](verification.json): происхождение цитаты, полный подсчёт,
  совпадение подготовок и неизменность исторических архивов.
- [review-protocol.md](review-protocol.md): прежние критерии четырёх случаев.
- Папки случаев: case/prepared/plan, фрагменты/ID/контекст/вычисление,
  затем либо answer/trace, либо error. Judgment при ошибке имеет no_answer.
- [metrics-before-review.json](metrics-before-review.json), [run.stdout.json](run.stdout.json):
  до смысловой оценки; [run.stderr.log](run.stderr.log): выполнение.
- [aliases-before.json](aliases-before.json): снимок не переиздавался.
- [tests-local.log](tests-local.log): 147 тестов, 144 прошли, 3 интеграционных пропущены.
- Данные: [12/10](validate-base.json), [40/20](validate-expanded.json),
  [40/4](validate-four.json).

Цитирование не доказывает релевантность или полноту ответа. Прежние данные,
индексы и архивы сохранены; новых зависимостей нет. Контракт и команды:
[docs/extractive.md](../../docs/extractive.md). Push не выполнялся.
