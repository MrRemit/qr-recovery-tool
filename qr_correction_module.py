# (Paste the full code from my previous response, August 15, 2025, 15:31 MST)
import math
import itertools
import logging
import sys
import re
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.FileHandler('qr_correction_log.txt', mode='w'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# Galois Field GF(2^8) utilities
class GF256:
    def __init__(self):
        self.prim_poly = 0x11D
        self.exp = [0] * 256
        self.log = [0] * 256
        self._build_tables()

    def _build_tables(self):
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= self.prim_poly
        self.exp[255] = self.exp[0]

    def add(self, a, b):
        return a ^ b

    def multiply(self, a, b):
        if a == 0 or b == 0:
            return 0
        return self.exp[(self.log[a] + self.log[b]) % 255]

    def divide(self, a, b):
        if a == 0:
            return 0
        if b == 0:
            raise ValueError("Division by zero")
        return self.exp[(self.log[a] - self.log[b] + 255) % 255]

    def inverse(self, a):
        if a == 0:
            raise ValueError("No inverse for 0")
        return self.exp[255 - self.log[a]]

gf = GF256()

def compute_syndrome(block, num_ecc):
    syndrome = [0] * num_ecc
    for i in range(num_ecc):
        for j, byte in enumerate(block):
            syndrome[i] = gf.add(syndrome[i], gf.multiply(byte, gf.exp[(i * j) % 255]))
    return syndrome

def berlekamp_massey_with_erasures(syndrome, erasure_positions, max_errors):
    n = len(syndrome)
    c = [0] * (n + 1)
    c[0] = 1
    b = [0] * (n + 1)
    b[0] = 1
    l = len(erasure_positions)
    m = 1
    erasure_poly = [1]
    for pos in erasure_positions:
        alpha = gf.exp[(255 - pos) % 255]
        erasure_poly = [gf.multiply(x, alpha) for x in erasure_poly] + [1]
    for i in range(len(erasure_poly)):
        c[i] = erasure_poly[i] if i < len(erasure_poly) else 0
    for k in range(n):
        delta = syndrome[k]
        for j in range(1, l + 1):
            if k - j >= 0:
                delta = gf.add(delta, gf.multiply(c[j], syndrome[k - j]))
        if delta != 0:
            t = c.copy()
            delta_inv = gf.inverse(delta) if delta != 0 else 1
            for j in range(n + 1):
                c[j] = gf.add(c[j], gf.multiply(b[j], delta))
            if 2 * l <= k + len(erasure_positions):
                l = k + 1 - l + len(erasure_positions)
                b = t
                m = k + 1
    return c[:l+1], l

def chien_search(error_locator, block_size, erasure_positions):
    error_positions = set(erasure_positions)
    for i in range(block_size):
        eval_sum = 0
        for j, coeff in enumerate(error_locator):
            eval_sum = gf.add(eval_sum, gf.multiply(coeff, gf.exp[(255 - i * j) % 255]))
        if eval_sum == 0:
            error_positions.add(i)
    return sorted(list(error_positions))

def forney_algorithm(syndrome, error_locator, error_positions):
    error_magnitudes = []
    omega = [0] * len(syndrome)
    for i in range(len(syndrome)):
        for j in range(len(error_locator)):
            if i + j < len(syndrome):
                omega[i] = gf.add(omega[i], gf.multiply(error_locator[j], syndrome[i + j]))
    for pos in error_positions:
        omega_eval = 0
        for i, coeff in enumerate(omega):
            omega_eval = gf.add(omega_eval, gf.multiply(coeff, gf.exp[(255 - pos * i) % 255]))
        lambda_deriv = 0
        for j in range(1, len(error_locator), 2):
            if j < len(error_locator):
                lambda_deriv = gf.add(lambda_deriv, gf.multiply(error_locator[j], gf.exp[(255 - pos * j) % 255]))
        magnitude = gf.divide(omega_eval, lambda_deriv) if lambda_deriv != 0 else 0
        error_magnitudes.append(magnitude)
    return error_magnitudes

def decode_qr_data(corrected_block, data_bytes):
    bits = ''.join(format(byte, '08b') for byte in corrected_block[:data_bytes])
    result = []
    pos = 0
    while pos < len(bits):
        if pos + 4 > len(bits):
            break
        mode = bits[pos:pos+4]
        pos += 4
        if mode == '0100':
            if pos + 8 > len(bits):
                break
            count = int(bits[pos:pos+8], 2)
            pos += 8
            if pos + count * 8 > len(bits):
                break
            data_bits = bits[pos:pos+count*8]
            data = ''
            for i in range(0, len(data_bits), 8):
                byte = int(data_bits[i:i+8], 2)
                data += chr(byte) if 32 <= byte <= 126 else '?'
            pos += count * 8
            result.append(data)
        elif mode == '0010':
            if pos + 9 > len(bits):
                break
            count = int(bits[pos:pos+9], 2)
            pos += 9
            if pos + count * 11 > len(bits):
                break
            data_bits = bits[pos:pos+count*11]
            alphanumeric = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:'
            data = ''
            for i in range(0, len(data_bits), 11):
                if i + 11 > len(data_bits):
                    break
                val = int(data_bits[i:i+11], 2)
                if val // 45 < len(alphanumeric) and val % 45 < len(alphanumeric):
                    data += alphanumeric[val // 45] + alphanumeric[val % 45]
                else:
                    data += '?'
            pos += count * 11
            result.append(data)
        else:
            break
    return ''.join(result)

def test_ecc_bytes(block, syndrome, num_ecc, data_bytes):
    results = []
    ecc_start = len(block) - num_ecc
    for r in range(1, min(4, num_ecc + 1)):
        for combo in itertools.combinations(range(ecc_start, len(block)), r):
            test_erasures = list(combo)
            result = correct_qr_block(block, test_erasures, num_ecc, data_bytes)
            if result['validation_score'] < 5 or result['decoded_string']:
                results.append({
                    'ecc_bytes': test_erasures,
                    'error_positions': result['error_positions'],
                    'error_magnitudes': result['error_magnitudes'],
                    'validation_score': result['validation_score'],
                    'decoded_string': result['decoded_string']
                })
    return results

def correct_qr_block(block, erasure_positions, num_ecc=30, data_bytes=40):
    max_errors = num_ecc // 2
    syndrome = compute_syndrome(block, num_ecc)
    if len(erasure_positions) + (len(syndrome) // 2) > max_errors:
        logger.info(f"Too many errors/erasures: {len(erasure_positions)} erasures + {len(syndrome)//2} errors > {max_errors}")
        return {
            'corrected_block': block,
            'validation_score': len(syndrome),
            'error_positions': [],
            'error_magnitudes': [],
            'decoded_string': ''
        }
    
    error_locator, error_count = berlekamp_massey_with_erasures(syndrome, erasure_positions, max_errors)
    error_positions = chien_search(error_locator, len(block), erasure_positions)
    error_magnitudes = forney_algorithm(syndrome, error_locator, error_positions)
    
    corrected_block = block.copy()
    for pos, mag in zip(error_positions, error_magnitudes):
        if pos < len(block):
            corrected_block[pos] = gf.add(block[pos], mag)
    
    new_syndrome = compute_syndrome(corrected_block, num_ecc)
    validation_score = sum(1 for s in new_syndrome if s != 0)
    decoded_string = decode_qr_data(corrected_block, data_bytes) if validation_score == 0 else ''
    
    return {
        'corrected_block': corrected_block,
        'validation_score': validation_score,
        'error_positions': error_positions,
        'error_magnitudes': error_magnitudes,
        'decoded_string': decoded_string
    }

def qr_array_to_block(qr_array, version=3):
    size = 29
    if len(qr_array) != size or any(len(row) != size for row in qr_array):
        raise ValueError("Invalid qr_array dimensions")
    
    block = []
    erasure_positions = []
    bit_string = ''
    bit_idx = 0
    
    # QR Version 3-L bit order (bottom-right, zigzag upward)
    coords = []
    for j in range(size-1, -1, -2):
        for i in range(size-1, -1, -1):
            for col in [j, j-1]:
                if col < 0:
                    continue
                if (i < 9 and col < 9) or (i < 9 and col >= size-8) or (i >= size-8 and col < 9):
                    continue
                coords.append((i, col))
    
    for i, j in coords:
        if bit_idx < 70 * 8:
            value = qr_array[i][j]
            if value not in (0, 1):
                bit_string += '?'
                if len(bit_string) == 8:
                    block.append(0)
                    erasure_positions.append(len(block)-1)
                    bit_string = ''
            else:
                bit_string += str(value)
                if len(bit_string) == 8:
                    block.append(int(bit_string, 2))
                    bit_string = ''
            bit_idx += 1
    
    if bit_string and len(bit_string) < 8:
        block.append(0)
        erasure_positions.append(len(block)-1)
    
    return block, erasure_positions

def compare_qr_arrays(reference_array, test_array):
    """
    Compare two 29x29 QR arrays and suggest module fixes.
    """
    if len(reference_array) != 29 or len(test_array) != 29 or any(len(row) != 29 for row in reference_array + test_array):
        raise ValueError("Invalid QR array dimensions")
    
    differences = []
    for i in range(29):
        for j in range(29):
            if reference_array[i][j] != test_array[i][j]:
                differences.append((i, j, reference_array[i][j], test_array[i][j]))
    
    return differences

def process_qr_array(qr_array, num_ecc=30, data_bytes=40):
    block, erasure_positions = qr_array_to_block(qr_array)
    result = correct_qr_block(block, erasure_positions, num_ecc, data_bytes)
    logger.info(f"QR Correction: Validation Score = {result['validation_score']}, Decoded = {result['decoded_string']}")
    
    if result['validation_score'] > 0:
        logger.info("\nTesting ECC byte combinations...")
        ecc_results = test_ecc_bytes(block, compute_syndrome(block, num_ecc), num_ecc, data_bytes)
        for res in ecc_results:
            logger.info(f"ECC Bytes {res['ecc_bytes']}:")
            logger.info(f"  Error Positions: {res['error_positions']}")
            logger.info(f"  Error Magnitudes: {res['error_magnitudes']}")
            logger.info(f"  Validation Score: {res['validation_score']}")
            logger.info(f"  Decoded String: {res['decoded_string']}")
            if res['validation_score'] == 0 and res['decoded_string']:
                result = res
    
    return result['corrected_block'], result['decoded_string'], result['validation_score']

if __name__ == "__main__":
    # Example 1 QR array (successful case)
    reference_qr_array = [
        [1 if c == '#' else 0 for c in line.strip()]
        for line in [
            "#######_______##_#_#__#######",
            "#_____#____#__####_#__#_____##",
            "#_###_#_#___#___#_#_#_#_###_#",
            "#_###_#__##_####__#___#_###_#",
            "#_###_#___#__#_#_##___#_###_#",
            "#_____#___##___##__#__#_____##",
            "#######_#_#_#_#_#_#_#_#######",
            "________##__##_#_#_##________",
            "###_#####_#__#_______##___#__",
            "_##_#______##__#_#___##____##",
            "____######__#_#_##___##_##_##",
            "##_##__#_#_###_#####___##____",
            "#_##_######___________#_##__#",
            "_####___##_###_#_#__#_#__####",
            "___#_####_#____###____###__##",
            "##_#____#_#__###_#__#_______#",
            "###_#_#######____#__####_#___",
            "___#_#__#_#_###_#_#_#_##_###_",
            "#___#_#####_###_###_###_#_###",
            "_#___#___#___#_###__#_###____",
            "#_##_###__#__#__###_######_##",
            "________##__#____#__#___#___#",
            "#######_###_##____###_#_#___#",
            "#_____#_#____#_#_#__#___##__#",
            "#_###_#_###______##_######___",
            "#_###_#___#__#____###____##__",
            "#_###_#_##__#_#___##___##_#_#",
            "#_____#_#__######_##_###___##",
            "#######_#_###______##_####_##"
        ]
    ]
    # Test with your QR array (replace with actual data)
    test_qr_array = reference_qr_array
    result = process_qr_array(test_qr_array, num_ecc=30, data_bytes=40)
