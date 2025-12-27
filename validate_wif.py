import base58, hashlib, sys

def verify(wif):
    try:
        decoded = base58.b58decode(wif)
        checksum = decoded[-4:]
        calculated = hashlib.sha256(hashlib.sha256(decoded[:-4]).digest()).digest()[:4]
        return checksum == calculated
    except:
        return False

if __name__ == "__main__":
    wif = sys.argv[1] if len(sys.argv) > 1 else "5Kagzxv8tX5Le1whNvWqwHXjGSzAJ8X6E9agZbFRnAD9vkTVXSF"
    print("VALID" if verify(wif) else "INVALID")
