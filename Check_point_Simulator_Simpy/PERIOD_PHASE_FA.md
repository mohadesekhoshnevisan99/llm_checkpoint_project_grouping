# انتخاب مشترک دوره و فاز با قید تازگی

فایل‌های بسته را در پوشهٔ simulator بگذارید؛ helperهای قبلی را با نسخه‌های داخل بسته جایگزین کنید. نیازی به dependency جدید نیست. برنامه‌های اصلی مخزن تغییر نمی‌کنند.

```bash
cd /home/mohadeseh/sampaper/Check_point_Simulator_Simpy
source .venv/bin/activate
python -m unittest test_period_phase -v
python optimize_period_phase.py --run results/grouped_baseline/boundary_before.json --periods 30 45 60 90 120 --phase-step 3 --max-age-s 120 --startup-grace-s 45 --network-gbps 8 --store-gbps 500 --seconds 10 --out-contract results/grouped_baseline/joint_candidate_contract.json --report results/grouped_baseline/joint_search.json
```

اگر status برابر `optimal_in_frozen_catalogue` یا `feasible_time_limit` بود و قرارداد جدید تولید شد، اجرا کنید:

```bash
python run_direct_contract.py --scenario scenarios/grouped_handcheck_no_failures.yaml --contract results/grouped_baseline/joint_candidate_contract.json --out results/grouped_baseline/joint_replay.json
python optimize_period_phase.py --run results/grouped_baseline/joint_replay.json --validate-replay --max-age-s 120 --startup-grace-s 45 --network-gbps 8 --store-gbps 500 --report results/grouped_baseline/joint_validation.json
```

اگر status ناممکن یا نامعلوم بود، هیچ قرارداد تازه‌ای نوشته نمی‌شود. قرارداد قدیمی موجود در همان مسیر نباید اجرا شود. برای retry مسیر خروجی تازه انتخاب کنید. `--seconds` محدودیت جست‌وجوی ترکیب‌هاست؛ ساخت گزینه‌ها قبل از شروع این زمان انجام می‌شود.

## مدل و دامنه

برای هر rank، حجم checkpoint و بیشترین offset شروع انتقال از مرز iteration و بیشترین مدت انتقال از trace استخراج می‌شود. برای هر job مجموعهٔ streamهای rankها بازسازی می‌شود: در handcheck چهار rank هرکدام یک GB در یک ثانیه دارند، پس موج آن job چهار GB/s است؛ نه یک GB/s. شکل موج روی هر capture پیش‌بینی‌شده انتقال داده می‌شود. بیشترین offset و duration فقط مشاهده‌های این trace هستند، نه کران معتبر برای هر سیاست یا بار آینده. نرخ یکنواخت size/duration همچنان تقریب است.

دوره‌های مجاز به همراه فازهای گام سه‌ثانیه‌ای بررسی می‌شوند. موعد به اولین مرز iteration مجاز نگاشت می‌شود. زمان کامل‌شدن نسخه آخرین پایان streamهای rankهاست. هر گزینه باید قید سن ۱۲۰ ثانیه و مهلت آغازین ۴۵ ثانیه را تا پایان job در trace مبنا رعایت کند. دیگر ثابت بودن تعداد checkpoint یا عدم افزایش فاصلهٔ آن‌ها شرط نیست؛ این تغییر لازمهٔ بهینه‌سازی frequency است. نرخ خرابی صفر باقی می‌ماند و هدف، کمترین حجم نوشتن با این قیود است، سپس کمترین overlap جفت انتقال‌ها. این هدف با کمینه‌سازی rollback موردانتظار یکی نیست.

جست‌وجوی branch-and-bound، یک گزینه برای هر job انتخاب و بودجهٔ شبکه/store مشترک را روی مجموع پروفایل rankها کنترل می‌کند. بنابراین این حل مستقل برای هر job نیست. `independent_bytes_lower_bound` کران پایین بدون رقابت بین jobهاست. `byte_gap_to_independent_bound` فاصله از آن کران است؛ یک کران ضعیف می‌تواند حتی در حل کامل فاصلهٔ مثبت داشته باشد. `optimal_in_frozen_catalogue` یعنی تمام جست‌وجوی لازم در این فهرست ثابت تمام شده، نه اثبات optimum واقعی simulator. هنگام time limit بهترین جواب مجاز یافته‌شده برگردانده می‌شود؛ نبود جواب در زمان محدود با infeasible بودن فرق دارد.

بودجهٔ ۸ GB/s یک محدودیت آزمایشی سخت‌تر برای جست‌وجو است، نه ظرفیت واقعی اندازه‌گیری‌شده و نه تغییر تنظیمات simulator. خود simulator هنوز از ظرفیت‌های native استفاده می‌کند. بازاجرا باید نشان دهد زمان‌ها/سن و نرخ‌های قابل استنتاج با این بودجه سازگار مانده‌اند. عدم رد بودجه در trace، اثبات اجرای محدودکنندهٔ لحظه‌ای نیست. `full_runtime_capacity_certified` همچنان false است.

مرحلهٔ فعلی فقط direct، گروه‌های تک‌عضوی، بدون خرابی و روی horizon محدود trace است. زمان پایان معلوم و حذف checkpoint در iteration آخر می‌تواند انتخاب فاز/تعداد موج را تحت تأثیر قرار دهد. برای آموزش طولانی یا arrivalهای جدید باید receding-horizon یا ارزیابی steady-state اضافه شود. افق استخراج‌شده در `evaluation_horizon_s` ثبت می‌شود؛ `horizon_s` قرارداد قدیمی در این runner به‌عنوان frame تکرارشونده استفاده نمی‌شود.

## نتیجهٔ آزمون محلی

۴۴ گزینهٔ متمایز برای هر job، ۱۴۴۰ گره جست‌وجو و حل کامل فهرست. با سقف ۸، کران مستقل ۳۶ GB بود ولی بهترین ترکیب مجاز ۴۰ GB شد؛ این نمونه اثر واقعی قید مشترک را نشان می‌دهد. دورهٔ هر سه job برابر ۹۰ و فازها alpha0=36، be0=36، bravo0=18 انتخاب شدند. فاز یکسان دو job مجاز است چون مجموع انتقال آن دو هشت GB/s است.

بازاجرای simulator: ۳ نسخه برای alpha0 و be0، چهار نسخه برای bravo0، مجموع ۴۰ GB، بیشترین سن نسخهٔ کامل حدود ۹۳٫۸۶ ثانیه و بدون نقض حد ۱۲۰. پایان آموزش ۳۱۸٫۲ در برابر ۳۱۸٫۶ قبلی شد؛ این تفاوت کوچکِ تک‌سناریو ادعای speedup عمومی نیست. کاهش حجم ۱۲۰→۴۰ با پذیرش نسخه‌های قدیمی‌تر (حدود ۴۰٫۸۹→۹۳٫۸۶ در age بیشینه) همراه است. قید آغازین ۴۵ نیز همچنان صرفاً فرض همین آزمایش است.

چهار آزمون: انتخاب دوره و فاز زیر بودجهٔ مشترک، رد age ناممکن، رد بودجهٔ کمتر از یک پروفایل job، و انتخاب/بازاجرای واقعی simulator. فایل‌های نسخهٔ اصلی تغییر نمی‌کنند. در نصب بدون simulator آزمون integration ممکن است skip شود؛ برای نتیجهٔ واقعی CLI را اجرا کنید.

مراحل باقی‌ماندهٔ هدف اصلی: کنترل واقعی budget یا ثبت نرخهای allocator، قرار دادن هزینهٔ خرابی و recovery در هدف، گروه‌های peer با محدودیت donor و assignment واقعی shard، و آزمایش مقیاس. این بسته صرفاً مرحلهٔ مشترک period/phase با تازگی در مسیر direct است.
