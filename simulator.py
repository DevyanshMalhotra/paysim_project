import requests
import time
import random
def make_random_tx():
    amt = round(random.uniform(1, 10000), 2)
    o = "C" + str(random.randint(100000, 999999))
    d = "M" + str(random.randint(100000, 999999))
    oldo = round(random.uniform(0, 20000), 2)
    newo = max(0, oldo - amt)
    oldd = round(random.uniform(0, 20000), 2)
    newd = oldd + amt
    t = random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
    return {
        "step": random.randint(1, 1000),
        "type": t,
        "amount": amt,
        "nameOrig": o,
        "oldbalanceOrg": oldo,
        "newbalanceOrig": newo,
        "nameDest": d,
        "oldbalanceDest": oldd,
        "newbalanceDest": newd
    }
url = "http://127.0.0.1:8000/predict"
for i in range(50):
    tx = make_random_tx()
    try:
        r = requests.post(url, json=tx, timeout=5)
        print(i, tx["amount"], r.text)
    except Exception as e:
        print("error", e)
    time.sleep(0.5)
