import logging
import base58
import numpy as np
import os
from datetime import datetime
from reedsolo import RSCodec, ReedSolomonError

# Setup logging
log_filename = f"qr_bitstream_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def log_info(message):
    logging.info(message)
    print(message)

def validate_wif(wif):
    try:
        decoded = base58.b58decode_check(wif)
        return len(decoded) in (37, 38)
    except:
        return False

def bitstream_to_bytes(bitstream):
    """Convert bit string to byte array"""
    return bytes(int(bitstream[i:i+8], 2) for i in range(0, len(bitstream), 8))

def bytes_to_bitstream(byte_data):
    """Convert byte array to bit string"""
    return ''.join(f"{byte:08b}" for byte in byte_data)

def repair_bitstream(bitstream, example_bitstream):
    """
    Repair QR bitstream by reconstructing mode indicators and CCIs
    based on example QR structure
    """
    # Extract mode indicators and CCIs from example bitstream
    example_segments = [
        {'mode': example_bitstream[0:4], 'type': '8-bit', 'length': int(example_bitstream[4:12], 2), 'cci_bits': 8},
        {'mode': example_bitstream[12:16], 'type': 'alphanumeric', 'length': int(example_bitstream[16:25], 2), 'cci_bits': 9},
        {'mode': example_bitstream[25:29], 'type': '8-bit', 'length': int(example_bitstream[29:37], 2), 'cci_bits': 8}
    ]
    
    total_header_bits = 37  # Sum of all mode+cci bits in example
    total_data_bits = sum([
        seg['length'] * (8 if seg['type'] == '8-bit' else 6)
        for seg in example_segments
    ])
    
    log_info(f"Original bitstream length: {len(bitstream)} bits")
    log_info(f"Example header bits: {example_bitstream[:37]}")
    
    # Apply Reed-Solomon error correction
    try:
        byte_data = bitstream_to_bytes(bitstream)
        corrected_data = RSCodec(15).decode(byte_data)[0]  # Version 3 has 15 ECC codewords
        corrected_bitstream = bytes_to_bitstream(corrected_data)
        log_info("Reed-Solomon correction applied successfully")
    except (ReedSolomonError, ValueError) as e:
        corrected_bitstream = bitstream
        log_info(f"Reed-Solomon correction failed: {str(e)}")
    
    # Repair mode indicators and CCIs
    ptr = 0
    repaired_bitstream = []
    decoded_segments = []
    wif_candidate = ""
    
    for i, segment in enumerate(example_segments):
        # Insert mode indicator
        mode_bits = segment['mode']
        if ptr + len(mode_bits) > len(corrected_bitstream):
            log_info(f"Warning: Not enough bits for segment {i} mode")
            break
            
        # Insert Character Count Indicator
        cci_bits = f"{segment['length']:0{segment['cci_bits']}b}"
        
        # Extract data bits
        if segment['type'] == '8-bit':
            data_bits_len = segment['length'] * 8
        elif segment['type'] == 'alphanumeric':
            data_bits_len = segment['length'] * 6
        else:
            data_bits_len = segment['length'] * 8  # Fallback
            
        if ptr + len(mode_bits) + segment['cci_bits'] + data_bits_len > len(corrected_bitstream):
            log_info(f"Warning: Not enough bits for segment {i} data")
            break
        
        # Extract data
        data_start = ptr + len(mode_bits) + segment['cci_bits']
        data_end = data_start + data_bits_len
        data_bits = corrected_bitstream[data_start:data_end]
        
        # Reconstruct segment
        repaired_bitstream.append(mode_bits)
        repaired_bitstream.append(cci_bits)
        repaired_bitstream.append(data_bits)
        
        # Decode data
        if segment['type'] == '8-bit':
            decoded = ''.join(chr(int(data_bits[i:i+8], 2)) 
                             for i in range(0, len(data_bits), 8))
        elif segment['type'] == 'alphanumeric':
            alnum_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
            decoded = ''.join(alnum_chars[int(data_bits[i:i+6], 2)] 
                             for i in range(0, len(data_bits), 6))
        else:
            decoded = data_bits
            
        decoded_segments.append(decoded)
        wif_candidate += decoded  # Append to WIF candidate
        log_info(f"Segment {i+1}: {segment['type']}, {segment['length']} chars")
        log_info(f"  Data: {decoded}")
        
        ptr = data_end
    
    # Combine remaining bits
    if ptr < len(corrected_bitstream):
        repaired_bitstream.append(corrected_bitstream[ptr:])
    
    full_bitstream = ''.join(repaired_bitstream)
    
    log_info(f"Repaired bitstream length: {len(full_bitstream)} bits")
    log_info(f"Full decoded content: {wif_candidate}")
    log_info(f"WIF candidate length: {len(wif_candidate)} characters")
    
    return full_bitstream, wif_candidate

def main():
    log_info("===== QR BITSTREAM RECOVERY STARTED =====")
    log_info("Using example QR structure for repair")
    
    # Example QR bitstream (from your good example)
    example_bitstream = (
        "0100" + "00011000" +  # 8-bit mode, 24 chars
        "0010" + "000001011" + # alphanumeric mode, 11 chars
        "0100" + "00010000" +  # 8-bit mode, 16 chars
        # Actual data would follow, but we only care about headers
        "0" * 386  # Placeholder for data bits
    )
    
    # Your damaged QR data blocks
    data_blocks=[ "01000001","10110011","01010100","10110110","00010110","01110111","10100111","10000111","01100011","10000111","01000101","10000011","01010100","11000110","01010011","00010111","01110110","10000100","11100111","01100110","01110111","00010011","01000100","10000111","10000110","10100100","01110100","01100111","10100010","00000011","10011101","01010011","00010010","01000111","00001001","01000001","00010110","00010110","01110101","10100110","10000100","01010101","00100110","11100100","00010100","11000011","00100111","00100110","10110101","01000101","01100101","10000101","00110100","01100000","11101010","00011011","10000000","00101011","11101001","10101110","11101110","10001011","11011111","11110000","00000100","01001001","00011011","10001111","00101100","00101101"]
    
    # Convert to single bitstream
    original_bitstream = ''.join(data_blocks)
    
    # Repair bitstream using example structure
    repaired_bitstream, wif_candidate = repair_bitstream(original_bitstream, example_bitstream)
    
    # Validate WIF
    if validate_wif(wif_candidate):
        log_info("WIF VALIDATION SUCCESSFUL!")
        log_info(f"Valid WIF: {wif_candidate}")
    else:
        log_info("WIF validation failed. Attempting targeted repairs...")
        
        # Apply Reed-Solomon error position
        error_position = 58  # 0-based index in byte array
        byte_position = error_position
        bit_index_in_byte = 0  # We'll test all bits in this byte
        
        # Convert to bit position
        bit_position = byte_position * 8 + bit_index_in_byte
        
        log_info(f"Applying error correction at position {byte_position} (bit {bit_position})")
        
        # Test all possible bit flips at the error position
        original_bit = repaired_bitstream[bit_position]
        flipped_bit = '1' if original_bit == '0' else '0'
        
        # Create repaired version
        repaired_bits = list(repaired_bitstream)
        repaired_bits[bit_position] = flipped_bit
        repaired_bitstream_flipped = ''.join(repaired_bits)
        
        # Decode flipped version
        _, wif_candidate_flipped = repair_bitstream(repaired_bitstream_flipped, example_bitstream)
        
        # Validate flipped version
        if validate_wif(wif_candidate_flipped):
            log_info("ERROR CORRECTION SUCCESSFUL WITH BIT FLIP!")
            log_info(f"Valid WIF: {wif_candidate_flipped}")
        else:
            log_info("Bit flip repair failed. Trying common character substitutions...")
            
            # Common repair patterns (position in string, not bit position)
            repair_patterns = [
                (57, 'K'),  # Position 58: X → K
                (57, 'k'),  # Position 58: X → k
                (56, 'B'),  # Position 57: 8 → B
                (56, '3'),  # Position 57: 8 → 3
                (58, 'S'),  # Position 59: 5 → S
                (58, 's'),  # Position 59: 5 → s
                (45, 'R'),  # Position 46: K → R
                (45, 'r'),  # Position 46: K → r
                (50, 'T'),  # Position 51: F → T
                (50, 't')   # Position 51: F → t
            ]
            
            for position, char in repair_patterns:
                repaired_wif = wif_candidate_flipped[:position] + char + wif_candidate_flipped[position+1:]
                if validate_wif(repaired_wif):
                    log_info(f"SUCCESSFUL REPAIR at position {position+1}: {char}")
                    log_info(f"Valid WIF: {repaired_wif}")
                    return
            
            log_info("No valid repair found")
            
            # Generate diagnostic report
            log_info("\n===== DIAGNOSTIC REPORT =====")
            log_info(f"Original WIF candidate: {wif_candidate}")
            log_info(f"Flipped WIF candidate: {wif_candidate_flipped}")
            log_info(f"Error position: {byte_position} (bit {bit_position})")
            log_info("Suggested next steps:")
            log_info("1. Verify QR version and error correction level match example")
            log_info("2. Check mode indicators in bitstream")
            log_info("3. Validate Character Count Indicators")
            log_info("4. Compare with known good QR structure:")
            log_info("   Segment 1: 8-bit, 24 chars")
            log_info("   Segment 2: alphanumeric, 11 chars")
            log_info("   Segment 3: 8-bit, 16 chars")
            log_info(f"5. Current decoded length: {len(wif_candidate)}")
            
            # Save bitstreams for inspection
            with open("original_bitstream.txt", "w") as f:
                f.write(original_bitstream)
            with open("repaired_bitstream.txt", "w") as f:
                f.write(repaired_bitstream_flipped)
            log_info("Saved bitstreams to original_bitstream.txt and repaired_bitstream.txt")

if __name__ == "__main__":
    main()
    log_info("===== RECOVERY PROCESS COMPLETED =====")
    log_info(f"Full log saved to {log_filename}")
