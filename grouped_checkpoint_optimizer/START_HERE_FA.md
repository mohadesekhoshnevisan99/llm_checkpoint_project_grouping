# اجرای گام‌به‌گام در Windows و VS Code

۱. ZIP را Extract All کن. در VS Code با File → Open Folder پوشه‌ای را باز کن که خود `optimizer.py` داخل آن است. برای نسخهٔ موجود این پوشه است:

`C:\Users\Khoshnevisan\Downloads\llm_checkpoint_project\experiments\grouped_checkpoint_optimizer`

۲. از Terminal → New Terminal یک PowerShell باز کن. دستور `Get-Location` باید همین پوشه را نشان دهد. این دستورها را به‌ترتیب بزن:

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s . -p "test_*.py" -v
```

Python 3.12 نسخهٔ استفاده‌شده در آزمون این بسته است. اگر `py -3.12` موجود نیست، ابتدا `py --list` را بررسی کن؛ اگر `python --version` نسخهٔ مناسب نصب‌شده را نشان داد، `python -m venv .venv` را به‌جای دستور ساخت محیط اجرا کن. اگر هیچ‌کدام شناخته نمی‌شوند، Python باید نصب شود. باقی دستورها همیشه با مسیر کامل Python داخل `.venv` هستند؛ فعال‌سازی محیط و تغییر ExecutionPolicy لازم نیست. نصب dependency به اینترنت نیاز دارد. پوشهٔ `.deps` مخصوص محیط این جلسه در ZIP نیست؛ محیط مجازی از آن استفاده نمی‌کند.

۳. برای قابلیت‌های Python در VS Code افزونه‌های Python و Python Debugger مایکروسافت را فعال/نصب کن. با Ctrl+Shift+P و Python: Select Interpreter مسیر `.venv\Scripts\python.exe` را انتخاب کن. برای اجرای terminal بالا افزونه لازم نیست؛ برای F5 لازم است.

۴. ابتدا همهٔ تست‌ها را اجرا کن. در پایان باید `Ran 10 tests` و `OK` ببینی. زمان اجرا ممکن است متفاوت باشد.

۵. مثال کوچک را اجرا کن:

```powershell
.\.venv\Scripts\python.exe optimizer.py --demo-jobs 12 --seconds 1 --output outputs/demo12.json
```

در Explorer فایل `outputs/demo12.json` را باز کن. `best` نباید null باشد و `best.feasible` باید true باشد. گروه‌ها در `best.groups` و تصمیم هر job در `best.policy` هستند. ورودی مصنوعی در `input` ذخیره می‌شود. انتخاب گروه‌ها در حالت timeout ممکن است بین اجراها متفاوت باشد.

۶. فایل `scenario.example.json` را باز کن؛ این مثال سه job دارد. آن را با Save As به `scenario.json` کپی کن و بعد داده‌های اندازه‌گیری‌شده را جایگزین کن:

```powershell
.\.venv\Scripts\python.exe optimizer.py --input scenario.example.json --seconds 1 --output outputs/example.json
.\.venv\Scripts\python.exe optimizer.py --input scenario.json --seconds 3 --output outputs/custom.json
```

دستور دوم فقط بعد از ساخت `scenario.json` اجرا می‌شود. مقدار `slot_s` گام زمانی است؛ همهٔ `periods_s` باید مضرب آن باشند و `horizon_s` بر تمام دوره‌ها بخش‌پذیر باشد. اندازه‌ها GB، نرخ‌ها GB/s و زمان‌ها ثانیه‌اند. `failure_rate_s` بر ثانیه است. شرح فیلدها در `GUIDE_FA.md` آمده است.

۷. تست مقیاس را اجرا کن:

```powershell
.\.venv\Scripts\python.exe optimizer.py --demo-jobs 300 --seconds 0 --output outputs/demo300.json
.\.venv\Scripts\python.exe summarize.py outputs/demo12.json outputs/demo300.json --output outputs/my_results.md
```

`--seconds 0` فقط heuristic است. مقدار مثبت بودجهٔ MILP برای هر گروه‌بندی است، نه کل برنامه. سقف پیش‌فرض اندازهٔ گروه ۸ است؛ الگوریتم از بین گزینه‌های ساخته‌شده انتخاب می‌کند، نه تمام افرازهای ممکن. این آزمایش synthetic جای اجرای simulator و اندازه‌گیری goodput را نمی‌گیرد.

۸. قرارداد policy برای توسعهٔ اتصال را خروجی بگیر:

```powershell
.\.venv\Scripts\python.exe policy_bridge.py --result outputs/demo12.json --output outputs/policy_contract.json
```

این فایل عمداً `integration_ready: false` دارد. آن را به `run_scenario.py --policy` نده. helperهای `next_capture_time` و `donor_allowed` تست شده‌اند ولی simulator هنوز آن‌ها را صدا نمی‌زند.

۹. برای debug فایل `optimizer.py` را باز کن؛ کنار خط داخل `solve_partition` breakpoint بگذار. Run and Debug → Optimizer: 12 jobs → F5. با F10 جلو برو و `groups`، `modes` و `caps` را ببین. فایل `.vscode/launch.json` فقط وقتی پوشهٔ خود نمونه را باز کرده‌ای مسیرهای درست دارد.

## ترتیب خواندن فایل‌ها

`scenario.example.json` → `optimizer.py` → `test_optimizer.py` → `policy_bridge.py` → `test_policy_bridge.py` → `GUIDE_FA.md`.

در optimizer به‌ترتیب `validate`، `partitions`، `catalogue`، `solve_partition` و `optimize` را بخوان. `catalogue` الگوی اشغال منابع برای گزینه‌های دوره/فاز/مسیر را می‌سازد. `solve_partition` همهٔ jobها را با قیود مشترک حل می‌کند؛ `optimize` گروه‌بندی‌ها را مقایسه می‌کند.

## اتصال به مخازن اصلی

با File → Add Folder to Workspace، مخازن `Check_point_Simulator_Simpy`، `checkpointing` و `control-plane` را برای خواندن اضافه کن. برای اجرای نمونه، terminal باید همچنان در پوشهٔ نمونه باشد. اضافه کردن پوشه‌ها به VS Code به‌تنهایی اتصال کد یا نصب dependency نیست.

مسیر دادهٔ موردنیاز:

```text
scenario اصلی + اندازه‌گیری‌ها
    → adapter ورودی با واحدها و شناسه‌های واقعی
    → optimizer.optimize(...)
    → policy_bridge.export_contract(...)
    → adapter اجرایی simulator (هنوز پیاده نشده)
    → capture timing + donor restrictions + route + global resource accounting
```

قدم‌های توسعه:

1. `gp_policy.py::build_model_inputs` را برای محاسبهٔ shard/rank و failure semantics بخوان. کلاس با job و job با نود یکسان نیست. adapter باید jobهای واقعی را expand و poolهای بدون هم‌پوشانی بسازد. sample `job-000` را به‌جای ID واقعی simulator استفاده نکن.
2. مسیر جداگانه‌ای مثلاً `grouped_policy_adapter.py` در simulator اضافه کن؛ policy جدید schema مجزا دارد. صرف rename کردن `period_s` به `checkpoint_every` صحیح نیست: اولی زمان است و دومی iteration.
3. در `run_scenario.py` بازوی آزمایشی جدید تعریف کن؛ بازوی قبلی baseline بماند. `run_crossjob.py::controller_epochs` نباید phaseهای policy جدید را در میانهٔ epoch دوباره جابه‌جا کند مگر کل policy بازحل شود.
4. در `checkpointing/crossjob.py` تمام مسیرهای رزرو donor، از جمله `_valid_donors`، `_reserve_probe`، `reserve_shards` و مسیرهای legacy، باید محدودیت مجاز را اعمال کنند. `donor_allowed` تنها predicate عضویت است، نه بررسی فضای آزاد/خرابی/ظرفیت.
5. زمان گرفتن snapshot تازه را با دوره و phase هر job هماهنگ کن. مسیر موجود از یک `slot_period` مشترک برای انتظار flush استفاده می‌کند؛ این با دوره‌های متفاوت capture در policy جدید معادل نیست. از helper `next_capture_time` فقط برای محاسبهٔ رویداد اسمی بعدی استفاده کن؛ readiness، شکست و missed slot هنوز وظیفهٔ simulator است. helper زمان strictly-after می‌دهد؛ رویداد اولیهٔ دقیقاً روی epoch را جدا schedule کن.
6. assignment کسری GB روی poolها را به shard/node واقعی تبدیل و دوباره ظرفیت را کنترل کن. route مستقیم به store، route peer و backstop باید صریح اجرا شوند. registry شبکه و store برای همهٔ گروه‌ها مشترک بماند.
7. قبل از خرابی، trace را بررسی کن: donor خارج گروه صفر؛ ظرفیت نقض‌شده صفر؛ interval/phase اجراشده مطابق policy؛ سپس خرابی source، donor و تغییر عضویت را تست کن. در تغییر گروه، recovery باید donorهای ثبت‌شدهٔ نسخه‌های قبلی را بشناسد.
8. تابع هزینهٔ MVP را با مدل چند-tier پروژه یکسان کن؛ سپس با baseline و seed یکسان goodput را مقایسه کن. prototype فعلی cadence مشترک peer/store و serialization کل گروه دارد، پس قیاس مستقیم آن با تمام قابلیت‌های سیستم اصلی عادلانه نیست.

این مراحل توضیح توسعهٔ لازم‌اند؛ اتصال خودکار به simulator در این بسته انجام نشده است. دستور ساختگی برای اجرای «روش جدید داخل simulator» ارائه نشده، چون چنین بازویی هنوز وجود ندارد.

## خطاهای معمول

- `No module named scipy/numpy`: دستور pip و اجرا باید هر دو با `.venv\Scripts\python.exe` باشند.
- فایل پیدا نمی‌شود: `Get-Location` و `Get-ChildItem` را ببین؛ باید `optimizer.py` در همان پوشه باشد.
- `best: null`: جواب مجاز پیدا نشده؛ پیام candidateها را بررسی کن. داده/RPO/ظرفیت را اصلاح یا دامنهٔ جست‌وجو را افزایش بده؛ infeasible بودن کاتالوگ با infeasible بودن کل سیستم یکی نیست.
- آزمون ۳۰۰ job کند است: نخست `--seconds 0`؛ سپس افزایش مرحله‌ای فازها و زمان MILP.
- در ZIP پوشهٔ outputs نیست: خروجی‌ها با اجرای دستورها ساخته می‌شوند. `RESULTS.md` گزارش اجرای قبلی است؛ برای اجرای خودت `outputs/my_results.md` را بخوان.
