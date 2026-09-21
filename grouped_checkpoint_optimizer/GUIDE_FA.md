# بهینه‌سازی مشترک گروه، فرکانس و فاز checkpoint

## پیشنهاد اصلی

گروه‌بندی را برای محدود کردن مجموعهٔ donorها و ساده‌تر کردن زمان‌بندی استفاده کنیم؛ تصمیم نهایی دربارهٔ فرکانس‌ها باید همچنان ظرفیت مشترک تمام گروه‌ها را ببیند. حل مستقل هر گروه با فرض دسترسی به کل شبکه، جواب سراسری معتبر نمی‌دهد.

الگوریتم پیشنهادی دو سطح دارد:

1. سطح بیرونی چند افراز از jobها می‌سازد؛ تعداد و اندازهٔ گروه‌ها ورودی ثابت نیست. ترکیب jobهای پرتقاضا با jobهای دارای ظرفیت donor بیشتر بررسی می‌شود.
2. برای هر افراز، یک حل سراسری، فاصلهٔ checkpoint، فاز و مسیر انتقال تمام jobها را انتخاب می‌کند. شبکه و object store محدودیت مشترک دارند. بهترین جواب مجاز بین افرازهای بررسی‌شده انتخاب می‌شود.

این روش بهینه‌سازی مشترک در مقیاس کلاستر است، ولی جست‌وجوی محدود افرازها تضمین optimum ریاضی روی تمام گروه‌بندی‌های ممکن نمی‌دهد. حتی حل دقیق مسئلهٔ داخلی فقط برای همان افراز، گام زمانی و فهرست گزینه‌ها بهینه است. عدد ۳۰۰ به‌تنهایی اثبات نمی‌کند حل بدون گروه غیرممکن است؛ باید زمان حل و کیفیت هر دو روش اندازه‌گیری شوند.

## ارتباط با کد موجود

مسیرهای زیر نسبت به مخازن هم‌سطح داخل Downloads هستند:

| کد | کار موجود | اقدام پیشنهادی |
|---|---|---|
| `checkpointing/multilevel/joint_async_optimizer.py` | حل مشترک فرکانس و سهم پهنای‌باند jobها | نگه‌داشتن به‌عنوان baseline و تولید حدود/گزینه‌های اولیه |
| `Check_point_Simulator_Simpy/gp_policy.py` | ساخت ورودی GP و تخصیص heuristic فاز بعد از حل | جایگزین کردن مسیر تصمیم‌گیری آزمایشی با policy جدید |
| `Check_point_Simulator_Simpy/run_crossjob.py`، تابع `controller_epochs` | جابه‌جایی فازها از روی مدت flush مشاهده‌شده | در نسخهٔ بعد، trigger بازحل مشترک با hysteresis |
| `Check_point_Simulator_Simpy/checkpointing/crossjob.py` | رزرو donor، انتقال، صف، recovery و ظرفیت مشترک | اعمال محدودیت گروه در تمام مسیرهای رزرو و کنترل اجرای policy |
| `checkpointing/multilevel/run_joint_controller.py` | نمونهٔ اتصال frequency optimization و زمان‌بندی integral | استفاده برای مقایسه؛ بدون وارد کردن مستقیم اسکریپت دارای اجرای top-level |
| `checkpointing/multilevel/run_congestion_integral.py` | زمان‌بندی انتقال در اسلات‌های واقعی | مرجع مقایسهٔ قیدهای زمانی |
| `control-plane/design/solved_phasing_spec.md` | طراحی لحاظ کردن اثر phasing روی فرکانس مؤثر | کمک به تعریف semantics؛ سند طراحی به‌تنهایی implementation نیست |
| `checkpointing/daemon/make_policy.py` | تبدیل interval به دوره‌های harmonic و offset | مسیر اتصال آینده به daemon پس از اعتبارسنجی simulator |

گروه‌بندی classها در بخش‌هایی از `gp_policy.py` برای فشرده‌سازی jobهای مشابه است؛ آن را نباید گروه واقعی با donorهای محدود فرض کرد. همچنین `run_congestion_optimize.py` با مدل fluid قدیمی deprecated شده؛ یکنواخت پخش کردن ترافیک در زمان می‌تواند برخورد واقعی انتقال‌ها را پنهان کند.

## مدل نمونهٔ قابل اجرا

فایل `optimizer.py` یک نمونهٔ مستقل است و به simulator یا daemon وصل نشده است. برای هر job یک گزینه انتخاب می‌شود:

`mode = (group, route, interval T, phase phi)`

فرکانس برابر `f = 1/T` است. زمان capture تازه برابر `phi + k*T` است؛ phase در این مدل زمان گرفتن نسخهٔ تازه را جابه‌جا می‌کند. این با گرفتن نسخه و سپس منتظر ماندن برای slot فرق دارد.

دو مسیر داریم:

- direct: capture → object store.
- peer: capture → SSD اعضای دیگر همان گروه → object store.

در مسیر peer، state به نسبت ظرفیت donorها تقسیم می‌شود؛ هیچ job به خودش اهدا نمی‌کند. دو نسخه روی donor رزرو می‌شود: قبلی و نسخهٔ در حال ساخت. همهٔ jobها در این نمونه از poolهای فیزیکی مجزا و همگن تشکیل شده‌اند. `donor_gbps` ظرفیت مؤثر اندازه‌گیری‌شدهٔ NIC و SSD آن pool است، نه سرعت اسمی دیسک. هر گروه در طول کل pipeline یک job را سرویس می‌دهد؛ این انتخاب محافظه‌کارانه تعارض sender/donor را حذف می‌کند اما هم‌زمانی مجاز سیستم واقعی را محدود می‌کند.

منظور از pairing در این نمونه رابطهٔ source→donor داخل گروه است، نه الزاماً جفت‌های یک‌به‌یک ثابت. اگر pairing دقیقاً matching یک‌به‌یک باشد، باید گزینه‌های donor به زیرمجموعه‌های مجاز محدود و assignment باینری اضافه شود؛ pooling فعلی معادل آن نیست. یک job چندنودی نیز با یک نود برابر نیست.

این نسخه یک cadence مشترک برای ارسال peer و store دارد. در معماری واقعی می‌توان `f_L2` و `f_L3` متفاوت داشت؛ قبل از ادعای بهبود نسبت به GP چندسطحی باید این تفاوت حذف یا کنترل شود. این نمونه parity، placement به‌ازای rank، تغییر عضویت آنلاین و خرابی واقعی حین اجرا را شبیه‌سازی نمی‌کند.

### تابع هدف

برای هر job با وزن `w`، هزینهٔ تقریبی زیر کمینه می‌شود:

```text
w * [ C/T + alpha*A/T
      + lambda_source*(T/2 + L_recoverable + R)
      + lambda_store*(T/2 + L_store + R) ]
```

`C` مدت capture، `A` زمان انتقال async، `R` restart، و lagها فاصلهٔ شروع capture تا کامل شدن نسخه‌اند. در peer، `L_recoverable` تا پایان نوشتن کامل peer است؛ در direct برابر lag کامل store است. همهٔ مدت‌ها به بالا به اسلات گرد می‌شوند.

این یک تقریب مرتبهٔ اول برای خرابی کم‌احتمال و ایستا است، نه اندازه‌گیری goodput و نه مدل دقیق renewal. `failure_rate_s` نرخ رخدادهایی است که source از دست می‌رود ولی peerهای لازم سالم می‌مانند. `store_failure_rate_s` نرخ رخدادهای مجزایی است که بازیابی به store نیاز دارد. این دو کلاس نباید دوباره‌شماری شوند. خرابی مستقل donor، failure domainهای هم‌بسته و از دست رفتن shard باید در مدل بعدی صریح شوند؛ این نسخه به‌تنهایی تضمین reliability عملیاتی ندارد.

`max_age_s` در صورت تعیین قید `T + L_recoverable` است؛ `max_store_age_s` قید `T + L_store` است. این‌ها age نسخهٔ قابل بازیابی در steady state هستند، نه زمان پایان recovery. شرط steady state مستلزم وجود checkpoint اولیهٔ معتبر است. مدل در warm-up و پس از خرابی دوباره باید ارزیابی شود.

### قیدهای سراسری

متغیر باینری `x[j,m]` مشخص می‌کند گزینهٔ m برای job j انتخاب شده است:

```text
sum_m x[j,m] = 1                                 for each job
sum_jm network[j,m,t] * x[j,m] <= B_network       for each slot t
sum_jm store[j,m,t] * x[j,m] <= B_store           for each slot t
sum_jm active_in_group[j,m,g,t] * x[j,m] <= 1     for each group and slot
sum_jm reserved_ssd[j,m,donor] * x[j,m] <= free_ssd[donor]
```

قید شبکه در نمونه مجموع payload همهٔ انتقال‌هاست. در شبکهٔ واقعی ممکن است ingress، egress، rack و link جداگانه محدود باشند؛ ظرفیت fabric را نباید بدون کالیبراسیون به این عدد تبدیل کرد. زمان‌بندی روی یک frame دوری است؛ انتقالی که از پایان frame عبور کند در ابتدای frame نیز ظرفیت می‌گیرد. دوره‌ها باید frame را تقسیم کنند. این انتخاب از انفجار LCM جلوگیری می‌کند، اما دامنهٔ فرکانس‌های قابل انتخاب را محدود می‌کند.

بدین ترتیب phase از مسیر feasibility روی فرکانس اثر می‌گذارد: اگر فازها با هم جا نشوند، حل‌کننده مجبور است دوره، مسیر یا افراز را تغییر دهد. یک جریمهٔ ساختگی برای phase اضافه نشده است.

## از صفر تا اجرای نمونه

### ۱. محیط مستقل بساز

در PowerShell، با Python نصب‌شده روی سیستم:

```powershell
cd C:\Users\Khoshnevisan\Downloads\llm_checkpoint_project\experiments\grouped_checkpoint_optimizer
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v
```

برای اجرای این جلسه SciPy در `.deps` نصب شده است؛ کد اگر آن پوشه موجود باشد از آن استفاده می‌کند. برای محیط مستقل تمیز، آن وابستگی محلی لازم نیست. `.deps` را وارد Git نکن.

### ۲. دادهٔ مصنوعی را اجرا کن

```powershell
.\.venv\Scripts\python.exe optimizer.py --demo-jobs 12 --seconds 3 --output outputs/demo12.json
.\.venv\Scripts\python.exe optimizer.py --demo-jobs 300 --seconds 0 --max-phases 12 --output outputs/demo300.json
```

`--seconds` محدودیت زمانی MILP **برای هر افراز** است؛ زمان کل شامل ساخت مدل و همهٔ افرازهاست. مقدار صفر فقط greedy و بهبود coordinate را اجرا می‌کند و تضمین optimum ندارد. `--max-group-size` سقف جست‌وجو است، نه اندازهٔ تحمیلی گروه. `--max-phases` تعداد گزینه‌های phase در هر دوره را محدود می‌کند. برای مسئلهٔ کوچک تمام اسلات‌های دوره را پوشش بده؛ برای مقیاس بزرگ حساسیت به این پارامتر را اندازه بگیر.

### ۳. ورودی واقعی را بساز

از یک فایل خروجی، بخش `input` را به `scenario.json` منتقل کن و مقادیر مصنوعی را با اندازه‌گیری جایگزین کن. همهٔ واحدها: ثانیه، GB و GB/s؛ نرخ خرابی بر ثانیه.

| فیلد | دادهٔ لازم |
|---|---|
| `size_gb` | اندازهٔ واقعی checkpoint pool پس از لحاظ DP/TP/PP/ZeRO |
| `capture_s` | وقفهٔ foreground اندازه‌گیری‌شده |
| `nic_gbps` | ظرفیت مؤثر ارسال source |
| `donor_gbps` | حداقل نرخ مؤثر دریافت/نوشتن و خواندن/ارسال donor |
| `free_ssd_gb` | ظرفیت قابل رزرو برای دیگر jobها، پس از کسر مصرف خود job |
| `failure_rate_s` | نرخ خرابی source با peer سالم |
| `store_failure_rate_s` | نرخ کلاس رخدادهای مجزای نیازمند store |
| `restart_s` | زمان restart/recovery تقریبی |
| `overlap` | افت نسبی محاسبه هنگام انتقال؛ باید کالیبره شود |
| `weight` | وزن job، مثلاً GPUهای تخصیص‌یافته برای هدف GPU-time |
| `network_gbps`, `store_gbps` | ظرفیت مشترک مؤثر کلاستر |
| `store_stream_gbps` | سقف ارسال یک pipeline به store |
| `headroom` | ضریب ظرفیت قابل استفاده، مثلاً ۰٫۸۵ |

```powershell
.\.venv\Scripts\python.exe optimizer.py --input scenario.json --seconds 5 --output outputs/real_input.json
```

هیچ نتیجهٔ synthetic را به‌عنوان نتیجهٔ workload واقعی گزارش نکن. capture یک‌ثانیه‌ای با slot پنج‌ثانیه‌ای در نمونه پنج ثانیه قیمت می‌گیرد؛ برای دادهٔ واقعی گام کوچک‌تر یا چند resolution را آزمایش کن.

برای بازتولید گزارش خلاصه از خروجی‌های ذخیره‌شده:

```powershell
.\.venv\Scripts\python.exe summarize.py outputs/demo12.json outputs/demo300.json
```

در اجرای این جلسه، ۱۲ job به چهار گروه سه‌عضوی و ۳۰۰ job به صد گروه سه‌عضوی رسیدند. زمان ثبت‌شده به‌ترتیب حدود ۴٫۸ و ۸٫۷ ثانیه بود؛ دومی با `--seconds 0` اجرا شد. در هر دو مثال هزینهٔ جواب به کران پایین تک-job همان افراز رسید، ولی تخصیص مستقل فازها قید گروه را نقض می‌کرد. بنابراین این مثال‌ها امکان یافتن فاز مجاز را نشان می‌دهند، نه بهبود goodput یا برتری frequency optimization روی دادهٔ واقعی. آزمون `test_global_store_couples_distinct_groups` جداگانه وضعیتی را می‌سنجد که کمبود ظرفیت مشترک واقعاً دورهٔ انتخابی را بلندتر می‌کند. جزئیات در `RESULTS.md` ثبت شده است.

### ۴. خروجی را درست تفسیر کن

`best.groups` تعداد و اعضای گروه‌های منتخب، و `best.policy` شامل دوره، فرکانس، فاز، route و تخصیص GB donorهاست. `peak_network_gbps` و `peak_store_gbps` باید پایین‌تر از ظرفیت ضربدر headroom باشند.

`optimal_within_solver_tolerance` فقط حل داخلی با tolerance یک‌هزارم است. `feasible_time_limit` جواب معتبر بدون اثبات کامل، `heuristic_feasible` جواب greedy، و `unknown` یعنی هنوز جواب معتبر پیدا نشده است. `infeasible_catalogue` فقط نبود جواب در همان فهرست محدود است؛ آن را به infeasible بودن سیستم واقعی تعمیم نده. `relative_gap_bound` فاصله از کران پایین همان افراز است و شامل کیفیت جست‌وجوی گروه‌ها نیست. وقتی `best` تهی است policy قابل اجرا تولید نشده؛ نباید آن را به daemon تحویل داد.

SciPy نتیجه، وضعیت و کران MILP را ارائه می‌کند؛ قرارداد آن در [مستندات رسمی milp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html) آمده است. GP پیوستهٔ قبلی را نمی‌توان صرفاً با اضافه کردن متغیر باینری به یک مسئلهٔ DGP معتبر تبدیل کرد؛ قواعد [DGP در CVXPY](https://www.cvxpy.org/tutorial/dgp/index.html) را باید رعایت کرد.

## اتصال مرحله‌ای به پروژهٔ اصلی

۱. **adapter ورودی**: یک تابع مستقل بنویس که scenario موجود را به این schema تبدیل کند. cohortهای simulator گاهی نمایندهٔ چند نود هستند؛ poolها نباید هم‌پوشانی فیزیکی داشته باشند. محاسبهٔ shard size را از مسیر موجود parallelism بگیر، نه از تعداد jobها. واحدها و failure semantics را با assertion کنترل کن.

۲. **خروجی policy**: برای هر job، `group_id`، donorهای مجاز، `interval_s`، `phase_s`، زمان شروع مشترک `epoch_origin`، نسخهٔ policy و expiry بساز. فرکانس wall-clock را با تقسیم و گرد کردن ساده به `checkpoint_every` تبدیل نکن مگر خطای زمان iteration اندازه‌گیری شده باشد.

۳. **اجرای phase**: در `crossjob.py` مسیر موجود می‌تواند snapshot گرفته‌شده را تا slot نگه دارد. نمونهٔ جدید capture تازه را نزدیک slot فرض می‌کند. یا trigger capture را جابه‌جا کن، یا زمان انتظار `Q` را به lag، RPO و حافظه اضافه و دوباره بهینه‌سازی کن. بدون این اصلاح، تابع هدف و اجرای واقعی هم‌معنا نیستند.

۴. **محدودیت donor**: فیلتر group را فقط در `_valid_donors` نگذار؛ `_reserve_probe`، `reserve_shards`، reservation قدیمی، fallback و recovery هم باید بررسی شوند. بازیابی نسخه‌های قدیمی باید حتی بعد از تغییر گروه ممکن بماند؛ metadata نسخه باید donorهای زمان ایجاد را حفظ کند. خروجی fractional این نمونه برای daemon نیازمند assignment واقعی shardها و بررسی ظرفیت هر نود است.

۵. **ظرفیت مشترک**: `CapacityRegistry` و صف‌های NIC/disk/store را global نگه دار. اگر برای هر گروه registry مستقل با کل ظرفیت بسازی، coupling سراسری از بین می‌رود.

۶. **multi-tier واقعی**: گزینه‌ها را به `(T2,T3,phi2,phi3,placement)` توسعه بده؛ lag، state retention و cost را با `AsyncModelSpec` موجود یکسان کن. تعداد گزینه‌ها را ابتدا با GP و dominance pruning کم کن. تنها پس از یکسان‌سازی مدل، ادعای بهبود نسبت به GP قبلی قابل دفاع است.

۷. **اجرای مقاوم به drift**: job arrival/departure، طولانی شدن flush و خرابی باید calendar را invalidate کند. سیاست fallback مشخص کن: حفظ آخرین نسخهٔ سالم، صف کنترل‌شده یا store fallback با بودجهٔ رزروشده؛ سپس بازحل با cooldown و hysteresis. تغییر گروه را تا drain نسخه‌های قدیمی atomic انجام نده؛ migration دورهٔ گذار لازم دارد.

۸. **کنترل‌پلین**: ابتدا offline simulator، سپس replay و shadow mode. مسیر ساخت policy فعلی و reload واقعی daemon باید جداگانه پیاده‌سازی و تست شوند؛ خواندن policy فقط در startup یک controller آنلاین نیست.

## آزمایش علمی لازم پیش از انتخاب نهایی

با همان workload، منابع، seed خرابی و semantics چهار baseline را مقایسه کن: فرکانس مستقل؛ GP سراسری فعلی؛ GP و phase پس‌پردازشی فعلی؛ روش مشترک جدید. donor-pool کامل را نیز با گروه‌بندی مقایسه کن تا هزینهٔ محدود کردن donorها روشن شود. برای ablation، فرکانس ثابت/فاز آزاد و فاز ثابت/فرکانس آزاد مفید است.

برای N برابر ۱۰، ۳۰، ۱۰۰ و ۳۰۰، بار کم تا اشباع، checkpointهای نامتوازن و خرابی source/donor/store اجرا کن. تعداد و اندازهٔ گروه منتخب، goodput واقعی، slowdown، rollback، صف p95/p99، store overflow، خطای فرکانس اجراشده، RPO violations، زمان حل و gap را گزارش کن. میانگین تابع هدف این prototype جای goodput نیست. چند seed و فاصلهٔ اطمینان لازم است.

اگر گروه‌بندی در بار کم سود نداشت، خاموش کردن peer/phasing باید یک گزینهٔ مجاز باقی بماند. اگر گروه‌های بزرگ لازم شدند، سقف جست‌وجو را بالا ببر؛ نتیجهٔ سقف ۸ دربارهٔ بهینگی گروه ۹ یا بیشتر چیزی نمی‌گوید. مرحلهٔ بعد برای کیفیت بهتر، split/merge و جابه‌جایی job بین گروه‌ها با ارزیابی دوبارهٔ حل سراسری است؛ در این MVP پیاده نشده است.

## دامنهٔ بررسی و تحویل

این راهنما و کد مستقل افزوده شده‌اند؛ مسیرهای عملیاتی سه مخزن تغییر نکرده‌اند. بررسی معماری، متن ارائه‌ها و مسیرهای اصلی در `PROJECT_UNDERSTANDING_2026-09-17.md` ثبت شده است. این کار ادعای ممیزی خط‌به‌خط همهٔ ۱۰۴ هزار خط، آزمون سخت‌افزاری یا اثبات نوآوری پژوهشی نیست.
