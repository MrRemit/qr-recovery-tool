# Save as ~/cuda_project/qr_recovery/check_wif.py
from bitcoinlib.keys import Key
wif = "5Kagzxv8tX5He1whNvgq4HxjGFzAJ8X6E9agZhERnAL2rkTVXSF"
try:
    key = Key(wif)
    address = key.address()
    print(f"Address: {address}")
    if address == "16L7tUpbuxwzqUt3A3y439hMyytJ34evDp":
        print("WIF is VALID for the address!")
    else:
        print("WIF is INVALID: Address mismatch")
except Exception as e:
    print(f"WIF is INVALID: {str(e)}")
