# اولین اتصال اجرایی: مسیر مستقیم، مرز iteration

`run_direct_contract.py` را در پوشهٔ simulator قرار دهید و با محیط Python همان simulator اجرا کنید. PyYAML و وابستگی‌های simulator کافی‌اند؛ SciPy برای این مرحله لازم نیست.

```bash
cd /home/mohadeseh/sampaper/Check_point_Simulator_Simpy
source .venv/bin/activate
python run_direct_contract.py --scenario scenarios/grouped_handcheck_no_failures.yaml --contract results/grouped_baseline/optimizer_policy_contract.json --out results/grouped_baseline/direct_contract_run.json
```

این فایل از `run_scenario.run_arm` و مسیر واقعی `store_mode` استفاده می‌کند. هیچ فایل simulator را ویرایش نمی‌کند. تنها در فرایند اجرای خودش، تابع حلقهٔ آموزش را با دو تغییر صریح و کنترل‌شده اجرا می‌کند: تصمیم checkpoint از موعد قرارداد می‌آید، و آموزش تا پایان capture همان موج منتظر می‌ماند؛ persist همچنان async است. تطبیق دقیق تکه‌های کد بررسی می‌شود؛ اگر نسخه متفاوت باشد با خطا متوقف می‌شود. جایگزینی تابع در پایان، حتی در صورت خطا، برگردانده می‌شود.

محدودهٔ مجاز: jobهای ثابت، یک worker به‌ازای rank، اندازهٔ shard صریح، بدون خرابی و preemption، فقط route=direct، گروه‌های تک‌عضوی و donorهای خالی. policy دارای peer با خطا رد می‌شود. این نسخه یک اجرای محدود برای اتصال است و جایگزین controller کامل گروه‌بندی نیست.

معنای زمان: اولین مرز پایان iteration بعد از موعد nominal. هیچ capture قدیمی برای رسیدن به slot نگه داشته نمی‌شود. اگر pipeline قبلی هنوز فعال باشد checkpoint عقب می‌افتد؛ موعدهای گذشته coalesce می‌شوند و شمارندهٔ skipped_deadlines ثبت می‌شود. در آخرین iteration checkpoint جدید ساخته نمی‌شود. فاز صفر پیش از اولین iteration اجرا نمی‌شود؛ نخستین مرز دارای state قابل ثبت استفاده می‌شود.

خروجی `decisions` زمان nominal، مرز تصمیم و iteration را ثبت می‌کند. بخش `audit.captures` زمان واقعی شروع capture هر rank را از trace می‌خواند. مقدار `max_capture_lag_s` شامل تأخیر مرز iteration و انتظار منابع capture است. trace واقعی gzip کنار فایل نتیجه با پسوند `.trace.jsonl.gz` ذخیره می‌شود. مسیرهای انتقال غیر از store و captureهای بدون تصمیم متناظر باعث خطای اعتبارسنجی می‌شوند.

ظرفیت‌های native simulator برای NIC و store مشترک باقی می‌مانند. بودجهٔ تجمیعی optimizer، سقف ارسال هر job و تقویم اسلاتی آن در این نسخه enforce نشده‌اند؛ سقف stream همان مقدار scenario است. به همین دلیل `optimizer_calendar_verified` برابر false است. `integration_ready` قرارداد اصلی تغییر نمی‌کند. این پرچم‌ها نباید دستی true شوند.

در آزمون محلی با سه job چهار‌نودی و دورهٔ ۳۰، فاز ۲۷: پایان آموزش ۳۱۸٫۶ ثانیه، ۳۰ موج job-level، ۱۲۰ capture در سطح rank، ۱۲۰ GB نوشتن store و بیشترین lag برابر ۱۲٫۷۲۹۷ ثانیه ثبت شد. این اعداد انتظار ثابت برای همهٔ نسخه‌ها نیستند. دو آزمون scheduler و اجرای واقعی simulator گذشتند.

baseline قبلی checkpoint_every=3 داشت، یعنی سه iteration شامل زمان compute و all-reduce؛ قرارداد جدید دورهٔ wall-clock برابر ۳۰ ثانیه دارد. بنابراین تعداد checkpoint برابر نیست و اختلاف زمان پایان اثبات بهبود الگوریتم نیست. قبل از مقایسهٔ علمی باید cadence، capture semantics، مدل ذخیره‌سازی و قیدهای منابع همسان شوند.

مراحل باقیمانده: تصمیم دربارهٔ اجرای دقیق phase یا مدل‌کردن lag مرز iteration؛ اعمال بودجه‌های مدل؛ اجرای گروه‌های peer و assignment واقعی shard؛ سپس افزودن failure/recovery و مقایسهٔ هم‌شرایط. این مراحل در فایل حاضر پیاده نشده‌اند.
