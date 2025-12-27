import math
import itertools
import logging
import sys
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('qr_correction_log.txt', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# Galois Field GF(2^8) utilities
class GF256:
    def __init__(self):
        self.prim_poly = 0x11D  # x^8 + x^4 + x^3 + x^2 + 1
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

def apply_mask_pattern(bits, mask_pattern, qr_version=3):
    """
    Apply QR code mask pattern to data bits.
    """
    size = 29  # QR Version 3 is 29x29
    masked_bits = ""
    bit_idx = 0
    for i in range(size):
        for j in range(size):
            if bit_idx >= len(bits):
                break
            # Skip function patterns (finder, alignment, etc.)
            if (i < 9 and j < 9) or (i < 9 and j >= size-8) or (i >= size-8 and j < 9):
                continue
            if mask_pattern == 0:
                mask = (i + j) % 2 == 0
            elif mask_pattern == 4:
                mask = (i * j) % 2 + (i * j) % 3 == 0
            else:
                mask = False
            masked_bits += '1' if (bits[bit_idx] == '0') ^ mask else '0'
            bit_idx += 1
    return masked_bits[:len(bits)]

def analyze_syndrome(syndrome):
    """
    Analyze syndrome to identify error characteristics.
    """
    syndrome_weight = sum(1 for s in syndrome if s != 0)
    critical_positions = [i for i, s in enumerate(syndrome) if s > 100]
    syndrome_entropy = -sum(
        (s/255 * math.log2(s/255) if s > 0 else 0) 
        for s in syndrome
    )
    return {
        'weight': syndrome_weight,
        'critical_positions': critical_positions,
        'entropy': syndrome_entropy
    }

def berlekamp_massey_with_erasures(syndrome, erasure_positions, max_errors):
    """
    Berlekamp-Massey algorithm with erasures in GF(2^8).
    """
    n = len(syndrome)
    c = [0] * (n + 1)  # Error locator polynomial
    c[0] = 1
    b = [0] * (n + 1)  # Auxiliary polynomial
    b[0] = 1
    l = len(erasure_positions)
    m = 1

    # Incorporate erasures
    erasure_poly = [1]
    for pos in erasure_positions:
        alpha = gf.exp[(255 - pos) % 255]
        erasure_poly = [gf.multiply(x, alpha) for x in erasure_poly] + [1]
    for i in range(len(erasure_poly)):
        c[i] = erasure_poly[i] if i < len(erasure_poly) else 0

    # Berlekamp-Massey for additional errors
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
    """
    Find error positions by evaluating error locator polynomial roots.
    """
    error_positions = set(erasure_positions)
    for i in range(block_size):
        eval_sum = 0
        for j, coeff in enumerate(error_locator):
            eval_sum = gf.add(eval_sum, gf.multiply(coeff, gf.exp[(255 - i * j) % 255]))
        if eval_sum == 0:
            error_positions.add(i)
    return sorted(list(error_positions))

def forney_algorithm(syndrome, error_locator, error_positions):
    """
    Compute error magnitudes using Forney algorithm.
    """
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

def compute_syndrome(block, num_ecc):
    """
    Compute syndrome for a block.
    """
    syndrome = [0] * num_ecc
    for i in range(num_ecc):
        for j, byte in enumerate(block):
            syndrome[i] = gf.add(syndrome[i], gf.multiply(byte, gf.exp[(i * j) % 255]))
    return syndrome

def decode_qr_data(corrected_block, data_bytes, mask_pattern=0):
    """
    Decode corrected block into QR data string.
    """
    bits = ''.join(format(byte, '08b') for byte in corrected_block[:data_bytes])
    # Skip masking for unmasked input blocks
    result = []
    pos = 0
    while pos < len(bits):
        if pos + 4 > len(bits):
            break
        mode = bits[pos:pos+4]
        pos += 4
        
        if mode == '0100':  # 8-bit Byte Mode
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
        elif mode == '0010':  # Alphanumeric Mode
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

def test_ecc_bytes(block, syndrome, num_ecc, data_bytes, mask_pattern):
    """
    Test ECC byte combinations for corrections.
    """
    results = []
    ecc_start = len(block) - num_ecc
    # Test single and double ECC bytes
    for r in range(1, 3):
        for combo in itertools.combinations(range(ecc_start, len(block)), r):
            test_erasures = list(combo)
            result = advanced_reed_solomon_correction(block, syndrome, test_erasures, num_ecc, data_bytes, mask_pattern)
            if result['validation_score'] < 5 or result['decoded_string']:
                results.append({
                    'ecc_bytes': test_erasures,
                    'error_positions': result['error_positions'],
                    'error_magnitudes': result['error_magnitudes'],
                    'validation_score': result['validation_score'],
                    'decoded_string': result['decoded_string']
                })
    return results

def advanced_reed_solomon_correction(block, syndrome, erasure_positions, num_ecc=30, data_bytes=40, mask_pattern=0):
    """
    Reed-Solomon correction with erasures for QR code data.
    """
    max_errors = num_ecc // 2
    if len(erasure_positions) + (len(syndrome) // 2) > max_errors:
        logger.info(f"Too many errors/erasures: {len(erasure_positions)} erasures + {len(syndrome)//2} errors > {max_errors}")
        return {
            'corrected_block': block,
            'validation_score': len(syndrome),
            'syndrome_analysis': analyze_syndrome(syndrome),
            'error_locator': [],
            'error_positions': [],
            'error_magnitudes': [],
            'decoded_string': ''
        }

    # Stage 1: Syndrome Analysis
    syndrome_analysis = analyze_syndrome(syndrome)
    logger.info(f"Syndrome Analysis: {syndrome_analysis}")
    
    # Stage 2: Error Locator Polynomial with Erasures
    error_locator, error_count = berlekamp_massey_with_erasures(syndrome, erasure_positions, max_errors)
    logger.info(f"Error Locator Polynomial: {error_locator}")
    
    # Stage 3: Find Error Positions
    error_positions = chien_search(error_locator, len(block), erasure_positions)
    logger.info(f"Error Positions: {error_positions}")
    
    # Stage 4: Compute Error Magnitudes
    error_magnitudes = forney_algorithm(syndrome, error_locator, error_positions)
    logger.info(f"Error Magnitudes: {error_magnitudes}")
    
    # Stage 5: Apply Corrections
    corrected_block = block.copy()
    for pos, mag in zip(error_positions, error_magnitudes):
        if pos < len(block):
            corrected_block[pos] = gf.add(block[pos], mag)
    
    # Stage 6: Validate by recomputing syndrome
    new_syndrome = compute_syndrome(corrected_block, num_ecc)
    validation_score = sum(1 for s in new_syndrome if s != 0)
    logger.info(f"Validation Score (Non-zero Syndromes): {validation_score}")
    
    # Stage 7: Decode to string
    decoded_string = decode_qr_data(corrected_block, data_bytes, mask_pattern) if validation_score == 0 else ''
    logger.info(f"Corrected Block (first 20 bytes): {corrected_block[:20]}")
    logger.info(f"Decoded String: {decoded_string}")
    
    # Log original vs corrected comparison
    logger.info("Original vs Corrected Comparison:")
    for i, (orig, corr) in enumerate(zip(block, corrected_block)):
        if orig != corr:
            logger.info(f"Byte {i}: Original {orig} → Corrected {corr}")
    
    return {
        'corrected_block': corrected_block,
        'validation_score': validation_score,
        'syndrome_analysis': syndrome_analysis,
        'error_locator': error_locator,
        'error_positions': error_positions,
        'error_magnitudes': error_magnitudes,
        'decoded_string': decoded_string
    }

def main():
    # Convert binary data blocks to bytes and recompute syndromes
    binary_data_l4 = [
        "01000001", "10110011", "01010100", "10110110", "00010110", "01110111", "10100111", "10000111",
        "01100011", "10000111", "01000101", "10000011", "01010100", "11000110", "01010011", "00010111",
        "01110110", "10000100", "11100111", "01100110", "01110111", "00010111", "01000100", "10000111",
        "10000110", "10100100", "01110100", "01100111", "10100010", "00000011", "10011101", "01010011",
        "00010010", "01000111", "00001001", "01000001", "00010110", "00010110", "01110101", "10100110",
        "10000100", "01010101", "00100110", "11100100", "00010100", "11000011", "00100111", "00100110",
        "10110101", "01000101", "01100101", "10000101", "00110100", "01100000", "11100010", "11010111",
        "01101100", "11100111", "???????/", "01011101", "11011101", "10111000", "11101100", "11000011",
        "???????/", "1000010?", "11010111", "01000011", "00011111", "???????/"
    ]
    erasure_positions_l4 = [58, 61, 63, 65]
    DATA_BLOCKS_L4 = []
    for i, b in enumerate(binary_data_l4):
        if re.search(r'[^01]', b):  # Check for non-binary characters (e.g., '?')
            DATA_BLOCKS_L4.append(0)
            if i not in erasure_positions_l4:
                erasure_positions_l4.append(i)
        else:
            DATA_BLOCKS_L4.append(int(b, 2))
    REED_SOLOMON_BLOCK_L4 = DATA_BLOCKS_L4
    SYNDROME_L4 = compute_syndrome(DATA_BLOCKS_L4, 30)
    ERASURE_POSITIONS_L4 = sorted(erasure_positions_l4)

    binary_data_l3_1 = [
        "01000001", "10110011", "01010100", "10110110", "00010110", "01110111", "10100111", "10000111",
        "01100011", "10000111", "01000101", "10000011", "01010100", "11000110", "01010011", "00010111",
        "01110110", "10000100", "11100111", "01100110", "01110111", "00010111", "01000100", "10000111",
        "10000110", "10100100", "01110100", "01100111", "10100010", "00000011", "10011101", "01010011",
        "00010010", "01000111", "00001001", "01000001", "00010110", "00010110", "01110101", "10100110",
        "10000100", "01010101", "00100110", "11100100", "00010100", "11000011", "00100111", "00100110",
        "10110101", "01000101", "01100101", "10000101", "00110100", "01100000", "11100010", "11010111",
        "01101100", "11100111", "???????/", "01011101", "11011101", "10111000", "11101100", "???????/",
        "00110111", "1000010?", "11010111", "01000011", "00011111", "01010110"
    ]
    erasure_positions_l3_1 = [58, 63, 65]
    DATA_BLOCKS_L3_1 = []
    for i, b in enumerate(binary_data_l3_1):
        if re.search(r'[^01]', b):
            DATA_BLOCKS_L3_1.append(0)
            if i not in erasure_positions_l3_1:
                erasure_positions_l3_1.append(i)
        else:
            DATA_BLOCKS_L3_1.append(int(b, 2))
    REED_SOLOMON_BLOCK_L3_1 = DATA_BLOCKS_L3_1
    SYNDROME_L3_1 = compute_syndrome(DATA_BLOCKS_L3_1, 30)
    ERASURE_POSITIONS_L3_1 = sorted(erasure_positions_l3_1)

    binary_data_l3_2 = [
        "01000001", "10110011", "01010100", "10110110", "00010110", "01110111", "10100111", "10000111",
        "01100011", "10000111", "01000101", "10000011", "01010100", "11000110", "01010011", "00010111",
        "01110110", "10000100", "11100111", "0110????", "????0111", "00010111", "01000100", "10000111",
        "10000110", "10100100", "01110100", "01100111", "10100010", "00000011", "10011101", "01010011",
        "00010010", "01000111", "00001001", "01000001", "00010110", "00010110", "01110101", "10100110",
        "10000100", "01010101", "00100110", "11100100", "00010100", "11000011", "00100111", "00100110",
        "10110101", "01000101", "01100101", "10000101", "00110100", "01100000", "11100010", "00011011",
        "???????/", "00101011", "11101001", "01101110", "11101110", "10001011", "11011111", "11110000",
        "00000101", "01001000", "00011011", "00001110", "00101000", "01101101"
    ]
    erasure_positions_l3_2 = [19, 20, 56]
    DATA_BLOCKS_L3_2 = []
    for i, b in enumerate(binary_data_l3_2):
        if re.search(r'[^01]', b):
            DATA_BLOCKS_L3_2.append(0)
            if i not in erasure_positions_l3_2:
                erasure_positions_l3_2.append(i)
        else:
            DATA_BLOCKS_L3_2.append(int(b, 2))
    REED_SOLOMON_BLOCK_L3_2 = DATA_BLOCKS_L3_2
    SYNDROME_L3_2 = compute_syndrome(DATA_BLOCKS_L3_2, 30)
    ERASURE_POSITIONS_L3_2 = sorted(erasure_positions_l3_2)

    binary_data_l2_1 = [
        "01000001", "10110011", "01010100", "10110110", "00010110", "01110111", "10100111", "10000111",
        "01100011", "10000110", "01000101", "10000011", "01010100", "11000110", "01010011", "00010111",
        "01110110", "10000100", "11100111", "01100111", "01110111", "00010111", "01000100", "10000111",
        "10000110", "10100100", "01110100", "01100111", "10100010", "00000011", "10011101", "01010011",
        "00010010", "01000111", "00001001", "01000001", "00010110", "00010110", "01110101", "10100110",
        "1000????", "????0101", "00100110", "11100100", "00010100", "10000011", "00110111", "00100110",
        "10110101", "01000101", "01100101", "10000101", "00110100", "01100000", "11100010", "00011011",
        "10100000", "00101011", "11101001", "01101110", "11101110", "10001011", "11011111", "11110000",
        "00000101", "01001000", "00011011", "00001110", "00101000", "00100101"
    ]
    erasure_positions_l2_1 = [40, 41, 67, 69]
    DATA_BLOCKS_L2_1 = []
    for i, b in enumerate(binary_data_l2_1):
        if re.search(r'[^01]', b):
            DATA_BLOCKS_L2_1.append(0)
            if i not in erasure_positions_l2_1:
                erasure_positions_l2_1.append(i)
        else:
            DATA_BLOCKS_L2_1.append(int(b, 2))
    REED_SOLOMON_BLOCK_L2_1 = DATA_BLOCKS_L2_1
    SYNDROME_L2_1 = compute_syndrome(DATA_BLOCKS_L2_1, 30)
    ERASURE_POSITIONS_L2_1 = sorted(erasure_positions_l2_1)

    binary_data_l2_2 = [
        "01000001", "10110011", "01010100", "10110110", "00010110", "01110111", "10100111", "10000111",
        "01100011", "10000110", "01000101", "10000011", "01010100", "11000110", "01010011", "00010111",
        "01110110", "10000100", "11100111", "01100111", "01110111", "00010111", "01000100", "10000111",
        "10000110", "10100100", "01110100", "01100111", "10100010", "00000011", "10011101", "01010011",
        "00010010", "01000111", "00001001", "01000001", "00010110", "00010110", "01110101", "10100110",
        "1000????", "????0101", "00100110", "11100100", "00010100", "10000011", "00110111", "00100110",
        "10110101", "01000101", "01100101", "10000101", "00110100", "01100000", "11100010", "00011011",
        "10100000", "00101011", "11101001", "01101110", "11101110", "10001011", "11011111", "11110000",
        "00000101", "01001000", "00011011", "00001110", "00101000", "00100101"
    ]
    erasure_positions_l2_2 = [40, 41, 67, 69]
    DATA_BLOCKS_L2_2 = []
    for i, b in enumerate(binary_data_l2_2):
        if re.search(r'[^01]', b):
            DATA_BLOCKS_L2_2.append(0)
            if i not in erasure_positions_l2_2:
                erasure_positions_l2_2.append(i)
        else:
            DATA_BLOCKS_L2_2.append(int(b, 2))
    REED_SOLOMON_BLOCK_L2_2 = DATA_BLOCKS_L2_2
    SYNDROME_L2_2 = compute_syndrome(DATA_BLOCKS_L2_2, 30)
    ERASURE_POSITIONS_L2_2 = sorted(erasure_positions_l2_2)

    # Version 3-M blocks (example, needs actual data)
    DATA_BLOCKS_M6 = [
        66, 166, 38, 151, 70, 54, 246, 150, 227, 163, 19, 100, 195, 119, 69, 87,
        6, 39, 87, 135, 119, 167, 21, 87, 67, 52, 19, 55, 147, 67, 51, 150, 132,
        215, 151, 151, 68, 163, 51, 70, 87, 100, 71, 0, 108, 170, 198, 44, 194,
        206, 248, 157, 110, 197, 162, 0, 154, 0, 123, 129, 72, 0, 46, 0, 156,
        136, 223, 54, 0, 0
    ]
    REED_SOLOMON_BLOCK_M6 = DATA_BLOCKS_M6
    SYNDROME_M6 = compute_syndrome(DATA_BLOCKS_M6, 26)
    ERASURE_POSITIONS_M6 = [55, 59, 61, 62, 64, 68]

    DATA_BLOCKS_M9 = [
        66, 166, 38, 151, 70, 54, 246, 150, 227, 163, 19, 100, 195, 119, 69, 87,
        0, 39, 87, 135, 119, 167, 21, 87, 67, 52, 19, 55, 147, 67, 51, 150, 132,
        0, 0, 151, 68, 163, 51, 70, 87, 100, 71, 0, 108, 170, 198, 44, 194,
        206, 248, 157, 110, 197, 162, 0, 154, 0, 123, 129, 72, 0, 46, 0, 156,
        136, 223, 54, 0, 0
    ]
    REED_SOLOMON_BLOCK_M9 = DATA_BLOCKS_M9
    SYNDROME_M9 = compute_syndrome(DATA_BLOCKS_M9, 26)
    ERASURE_POSITIONS_M9 = [33, 34, 55, 59, 61, 62, 64, 68, 69]

    DATA_BLOCKS_M11 = [
        66, 166, 38, 151, 70, 54, 246, 150, 227, 163, 19, 100, 195, 119, 69, 87,
        0, 39, 87, 135, 119, 167, 21, 0, 0, 52, 19, 55, 147, 67, 51, 150, 132,
        0, 0, 151, 68, 163, 51, 70, 0, 100, 71, 0, 108, 170, 198, 44, 194,
        206, 248, 157, 110, 197, 162, 0, 154, 0, 123, 129, 72, 0, 46, 0, 0,
        136, 223, 54, 0, 0
    ]
    REED_SOLOMON_BLOCK_M11 = DATA_BLOCKS_M11
    SYNDROME_M11 = compute_syndrome(DATA_BLOCKS_M11, 26)
    ERASURE_POSITIONS_M11 = [16, 23, 24, 33, 34, 40, 55, 59, 61, 62, 64]

    NUM_ECC_L = 30
    DATA_BYTES_L = 40
    MASK_PATTERN_L = 0
    NUM_ECC_M = 26
    DATA_BYTES_M = 44
    MASK_PATTERN_M = 4

    # Run correction for all QR data
    configs = [
        ("Version 3-L (4 Erasures)", REED_SOLOMON_BLOCK_L4, SYNDROME_L4, ERASURE_POSITIONS_L4, NUM_ECC_L, DATA_BYTES_L, MASK_PATTERN_L),
        ("Version 3-L (3 Erasures, First)", REED_SOLOMON_BLOCK_L3_1, SYNDROME_L3_1, ERASURE_POSITIONS_L3_1, NUM_ECC_L, DATA_BYTES_L, MASK_PATTERN_L),
        ("Version 3-L (3 Erasures, Second)", REED_SOLOMON_BLOCK_L3_2, SYNDROME_L3_2, ERASURE_POSITIONS_L3_2, NUM_ECC_L, DATA_BYTES_L, MASK_PATTERN_L),
        ("Version 3-L (2 Erasures, First)", REED_SOLOMON_BLOCK_L2_1, SYNDROME_L2_1, ERASURE_POSITIONS_L2_1, NUM_ECC_L, DATA_BYTES_L, MASK_PATTERN_L),
        ("Version 3-L (2 Erasures, Second)", REED_SOLOMON_BLOCK_L2_2, SYNDROME_L2_2, ERASURE_POSITIONS_L2_2, NUM_ECC_L, DATA_BYTES_L, MASK_PATTERN_L),
        ("Version 3-M (6 Erasures)", REED_SOLOMON_BLOCK_M6, SYNDROME_M6, ERASURE_POSITIONS_M6, NUM_ECC_M, DATA_BYTES_M, MASK_PATTERN_M),
        ("Version 3-M (9 Erasures)", REED_SOLOMON_BLOCK_M9, SYNDROME_M9, ERASURE_POSITIONS_M9, NUM_ECC_M, DATA_BYTES_M, MASK_PATTERN_M),
        ("Version 3-M (11 Erasures)", REED_SOLOMON_BLOCK_M11, SYNDROME_M11, ERASURE_POSITIONS_M11, NUM_ECC_M, DATA_BYTES_M, MASK_PATTERN_M)
    ]

    # Summary for significant results
    summary = []

    for name, block, syndrome, erasures, num_ecc, data_bytes, mask_pattern in configs:
        logger.info(f"\n--- {name} ---")
        result = advanced_reed_solomon_correction(block, syndrome, erasures, num_ecc, data_bytes, mask_pattern)
        if result['validation_score'] == 0 or result['decoded_string']:
            summary.append({
                'name': name,
                'validation_score': result['validation_score'],
                'decoded_string': result['decoded_string'],
                'error_positions': result['error_positions']
            })
        
        # Debug ECC bytes
        logger.info(f"\n--- ECC Byte Debugging for {name} ---")
        ecc_results = test_ecc_bytes(block, syndrome, num_ecc, data_bytes, mask_pattern)
        for res in ecc_results:
            logger.info(f"ECC Bytes {res['ecc_bytes']}:")
            logger.info(f"  Error Positions: {res['error_positions']}")
            logger.info(f"  Error Magnitudes: {res['error_magnitudes']}")
            logger.info(f"  Validation Score: {res['validation_score']}")
            logger.info(f"  Decoded String: {res['decoded_string']}")
            if res['validation_score'] == 0 or res['decoded_string']:
                summary.append({
                    'name': f"{name} (ECC Debug {res['ecc_bytes']})",
                    'validation_score': res['validation_score'],
                    'decoded_string': res['decoded_string'],
                    'error_positions': res['error_positions']
                })

    # Log summary
    logger.info("\n--- Summary of Significant Results ---")
    for item in summary:
        logger.info(f"{item['name']}:")
        logger.info(f"  Validation Score: {item['validation_score']}")
        logger.info(f"  Decoded String: {item['decoded_string']}")
        logger.info(f"  Error Positions: {item['error_positions']}")

if __name__ == "__main__":
    main()
