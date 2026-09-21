# محدودکنندهٔ واقعی ظرفیت مشترک در simulator

فایل‌های این بسته را در پوشهٔ `Check_point_Simulator_Simpy` قرار دهید. `run_direct_contract.py` را با نسخهٔ همراه جایگزین کنید. منبع simulator و capacity.py ویرایش نمی‌شوند. hookها فقط در فرایند اجرای این برنامه فعال‌اند و حتی هنگام خطا restore می‌شوند.

```bash
cd /home/mohadeseh/sampaper/Check_point_Simulator_Simpy
source .venv/bin/activate
python -m unittest test_runtime_budget -v
python run_budget_contract.py --scenario scenarios/grouped_handcheck_no_failures.yaml --contract results/grouped_baseline/joint_candidate_contract.json --out results/grouped_baseline/joint_budget_replay.json --network-gbps 8 --store-gbps 500
python evaluate_durability.py --run results/grouped_baseline/joint_budget_replay.json --max-age-s 120 --startup-grace-s 45 --network-gbps 8 --store-gbps 500 --out results/grouped_baseline/joint_budget_age.json
```

سه تست باید بگذرد: رد بودجهٔ نامعتبر، کنترل انتگرال نرخ/حجم، و تست واقعی دو گلوگاه شبکه/store. تست سوم از قرارداد هم‌فاز ۲۷ ثانیه‌ای با تقاضای ۱۲ GB/s استفاده می‌کند. با سقف شبکهٔ ۸، زمان انتقال هم‌زمان rankها به ۱٫۵ ثانیه می‌رسد؛ با سقف store برابر ۴ به سه ثانیه می‌رسد. در هر دو حالت حجم کل ۱۲۰ GB باقی می‌ماند.

برنامهٔ جدید روی هر stream به store، دو منبع مجازی مشترک اضافه می‌کند: بودجهٔ checkpoint شبکه و بودجهٔ checkpoint store. allocator موجود max-min fair این منابع را هم‌زمان با NIC و store واقعی لحاظ می‌کند. داشتن ظرفیت مجزا برای هر job اتفاق نمی‌افتد؛ همهٔ jobها از همان منابع مجازی استفاده می‌کنند.

پس از هر تغییر تخصیص event-engine، نرخ واقعی allocator ثبت می‌شود. نرخ‌ها بین رویدادها ثابت‌اند. نمونه‌های هم‌زمان به آخرین وضعیت آن لحظه ادغام می‌شوند و قبل از ادغام رعایت سقف بررسی می‌شود. انتگرال نرخ شبکه در زمان باید با bytes تکمیل‌شده در store برابر باشد. نمونه‌ها همراه نرخ هر stream در `runtime_budget_audit.allocation_samples` ذخیره می‌شوند. این ثبت با تخمین size/duration متفاوت است.

در اجرای محلی قرارداد منتخب، peak شبکه و store هر دو ۸ GB/s، انتگرال شبکه ۴۰ GB، و حجم store نیز ۴۰ GB بودند. مقدار `runtime_checkpoint_budget_verified` برابر true شد. سن نسخهٔ کامل همچنان حدود ۹۳٫۸۶ ثانیه و پایان آموزش ۳۱۸٫۲ ثانیه بود. این نتیجه به همین سناریو/قرارداد محدود است.

فیلد جدید داخل `runtime_budget_audit` تأیید محدود ظرفیت checkpoint است؛ نه اثبات عمومی سیاست. `optimizer_calendar_verified` همچنان false است چون capture روی مرز iteration رخ می‌دهد. گزارش قدیمی evaluate_durability همچنان `instantaneous_capacity_verified:false` می‌دهد چون خودش فقط byte/time را می‌سنجد؛ برای شاهد نرخ واقعی، بخش runtime_budget_audit را بخوانید. این دو فیلد روش‌های بررسی متفاوت دارند و نباید دستی تغییر داده شوند.

دامنهٔ این مرحله: direct-store، event engine، بدون خرابی، بدون background traffic، گروه‌های تک‌عضوی. بودجه برای payload checkpoint است؛ ترافیک training، سربار پروتکل، مسیر peer، بازیابی و fabric topology چندلینکی را تضمین نمی‌کند. polling با خطا رد می‌شود. محدودیت store مجازی برای همین checkpointهاست، نه همهٔ کاربران ممکن یک object store واقعی.

بودجهٔ ۸ همچنان پارامتر آزمایشی انتخاب‌شده است، نه ظرفیت اندازه‌گیری‌شدهٔ کلاستر. فرق این مرحله آن است که حالا این پارامتر در runtime واقعاً enforce می‌شود و انتقال کند خواهد شد. برای budgetهای کمتر، پس از هر اجرا age را دوباره بررسی کنید؛ محدودکننده به‌خودی‌خود قید freshness را تضمین نمی‌کند.

برای بررسی دستی آزمایش فشار روی قرارداد قدیمی:

```bash
python run_budget_contract.py --scenario scenarios/grouped_handcheck_no_failures.yaml --contract results/grouped_baseline/optimizer_policy_contract.json --out results/grouped_baseline/old_contract_budget8.json --network-gbps 8 --store-gbps 500
```

این اجرا لازم نیست مگر بخواهید رفتار throttling را جدا ببینید؛ تست خودکار آن را پوشش می‌دهد. مرحلهٔ بعد هدف اصلی، توسعه به peer با محدودیت donor داخل گروه، و پس از آن اضافه‌کردن failure/recovery است.
