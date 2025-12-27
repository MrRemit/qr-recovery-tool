import numpy as np
import base58
from reedsolo import RSCodec, ReedSolomonError
from itertools import combinations
import time

class QRBruteForceRecovery:
    def __init__(self, data_blocks, version=3, ec_level='L'):
        self.version = version
        self.ec_level = ec_level
        self.data_blocks = data_blocks
        self.ecc_capacity = 15  # Version 3, Level L
        self.total_bytes = len(data_blocks)
        self.critical_bytes = self.identify_critical_bytes()
        self.expected_prefix = "5Kag"
        self.expected_suffix = "XSF"
        self.expected_length = 51

    def identify_critical_bytes(self):
        """Focus on high-risk byte positions instead of individual bits"""
        return [
            60, 61, 62, 63, 64, 65,  # Critical region around erasure
            66, 67, 68, 69,           # Adjacent bytes
            15, 20, 23, 42, 55        # Positions from your manual tests
        ]

    def bytes_to_bitstream(self, byte_array):
        return ''.join(f'{b:08b}' for b in byte_array)

    def decode_bitstream(self, bitstream):
        try:
            byte_array = [int(bitstream[i:i+8], 2) 
                          for i in range(0, len(bitstream), 8)]
            corrected = RSCodec(self.ecc_capacity).decode(byte_array)[0]
            return corrected.decode('utf-8', errors='ignore')
        except (ReedSolomonError, UnicodeDecodeError):
            return None

    def validate_wif(self, candidate):
        if not candidate or len(candidate) != self.expected_length:
            return False
        try:
            decoded = base58.b58decode_check(candidate)
            return len(decoded) in (37, 38)
        except ValueError:
            return False

    def batch_test_combinations(self, base_bytes, byte_positions, max_flips=2, progress_callback=None):
        """Efficiently test byte combinations in batches

        Args:
            base_bytes: Base byte array to modify
            byte_positions: Positions to try modifying
            max_flips: Maximum number of bytes to flip simultaneously
            progress_callback: Optional callback function for progress updates
        """
        # Create base byte array
        base_arr = np.array(base_bytes, dtype=np.uint8)

        # Generate all possible flip combinations
        for flip_count in range(1, max_flips + 1):
            print(f"Testing {flip_count}-byte combinations...")
            for byte_combo in combinations(byte_positions, flip_count):
                for value_combo in self.generate_byte_values(flip_count):
                    test_bytes = base_arr.copy()

                    # Apply byte modifications
                    for i, byte_pos in enumerate(byte_combo):
                        test_bytes[byte_pos] = value_combo[i]

                    # Convert to bitstream and decode
                    bitstream = self.bytes_to_bitstream(test_bytes)
                    decoded = self.decode_bitstream(bitstream)

                    # Update progress if callback provided
                    if progress_callback:
                        progress_callback()

                    if decoded and decoded.startswith(self.expected_prefix) and \
                       decoded.endswith(self.expected_suffix) and \
                       self.validate_wif(decoded[:self.expected_length]):
                        return decoded[:self.expected_length]
        return None

    def generate_byte_values(self, count):
        """Generate batches of possible byte values"""
        # First batch: try values close to original
        for i in range(count):
            for value in range(256):
                yield [value] * count
                
        # Second batch: try common error patterns
        error_patterns = [0x00, 0xFF, 0x55, 0xAA, 0x0F, 0xF0, 0x33, 0xCC]
        for pattern in error_patterns:
            yield [pattern] * count

    def smart_recovery(self):
        """Optimized recovery with batched testing"""
        print("Starting smart recovery...")
        
        # Convert data blocks to integers
        base_bytes = []
        for block in self.data_blocks:
            if '?' in block:
                base_bytes.append(int(block.replace('?', '0'), 2))
            else:
                base_bytes.append(int(block, 2))
        
        # Phase 1: Single-byte modifications
        print("\n[Phase 1] Testing single-byte modifications")
        result = self.batch_test_combinations(base_bytes, self.critical_bytes, max_flips=1)
        if result:
            return result
        
        # Phase 2: Double-byte modifications
        print("\n[Phase 2] Testing double-byte combinations")
        return self.batch_test_combinations(base_bytes, self.critical_bytes, max_flips=2)

if __name__ == "__main__":
    # Your data blocks
    data_blocks = [
        "01000001", "10110011", "01010100", "10110110", "00010110", "01110111",
        "10100111", "10000111", "01100011", "10000111", "01000101", "10000011",
        "01010100", "11000110", "01010011", "00010111", "01110110", "10000100",
        "11100111", "01100110", "01110111", "00010011", "01000100", "10000111",
        "10000110", "10100100", "01110100", "01100111", "10100010", "00000011",
        "10011101", "01010011", "00010010", "01000111", "00001001", "01000001",
        "00010110", "00010110", "01110101", "10100110", "10000100", "01010101",
        "00100110", "11100100", "00010100", "11000011", "00100111", "00100110",
        "10110101", "01000101", "01100101", "10000101", "00110100", "01100000",
        "11100010", "00011011", "10100000", "00101011", "11101001", "11101110",
        "11101110", "10001011", "11011111", "11110000", "00000101", "01001001",
        "00011011", "00001110", "00101000", "00100101"
    ]

    start_time = time.time()
    recovery = QRBruteForceRecovery(data_blocks)
    result = recovery.smart_recovery()
    
    if result:
        print(f"\nSUCCESS! Recovered WIF: {result}")
    else:
        print("\nRecovery failed after exhaustive search")
    
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
