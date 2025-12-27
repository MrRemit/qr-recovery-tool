#!/usr/bin/env python3
"""
QR Paper Wallet Recovery - Command Line Interface
User-friendly CLI with progress tracking
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from qr_recovery import QRBruteForceRecovery
from qr_image_processor import extract_from_image, analyze_qr_damage
import time

def print_banner():
    """Display tool banner"""
    print("=" * 70)
    print("🔐 QR Paper Wallet Recovery Tool")
    print("   Advanced Reed-Solomon ECC + Intelligent Brute Force")
    print("=" * 70)
    print()

def main():
    parser = argparse.ArgumentParser(
        description='Recover damaged Bitcoin paper wallet QR codes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic recovery from image
  python qr_cli.py wallet.jpg

  # Specify QR version and max error corrections
  python qr_cli.py wallet.jpg --version 3 --max-flips 4

  # Use GPU acceleration
  python qr_cli.py wallet.jpg --gpu

  # Verbose output with detailed logging
  python qr_cli.py wallet.jpg -v
        """
    )

    parser.add_argument('image', type=str, help='QR code image file path')
    parser.add_argument('--version', type=int, default=3,
                       help='QR code version (default: 3)')
    parser.add_argument('--ec-level', type=str, default='L',
                       choices=['L', 'M', 'Q', 'H'],
                       help='Error correction level (default: L)')
    parser.add_argument('--max-flips', type=int, default=3,
                       help='Maximum bit flips to attempt (default: 3)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration (requires CUDA)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for recovered key')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bars')

    args = parser.parse_args()

    # Validate input file
    if not Path(args.image).exists():
        print(f"❌ Error: Image file '{args.image}' not found!")
        sys.exit(1)

    print_banner()

    # Step 1: Extract QR data from image
    print("📷 Step 1/4: Scanning QR code from image...")
    try:
        qr_data, qr_info = extract_from_image(args.image)
        if qr_data is None:
            print("❌ Failed to read QR code from image")
            print("   Try:")
            print("   - Better lighting/contrast")
            print("   - Higher resolution scan")
            print("   - Manual transcription")
            sys.exit(1)

        print(f"   ✓ QR Code detected: {qr_info['type']}")
        print(f"   ✓ Data length: {len(qr_data)} bytes")
    except Exception as e:
        print(f"❌ Error scanning image: {e}")
        sys.exit(1)

    # Step 2: Analyze damage
    print("\n🔍 Step 2/4: Analyzing damage level...")
    damage_info = analyze_qr_damage(qr_data, args.version, args.ec_level)

    print(f"   • Estimated damage: {damage_info['damage_percent']:.1f}%")
    print(f"   • Strategy: {damage_info['strategy']}")
    print(f"   • Estimated time: {damage_info['estimated_time']}")

    if damage_info['damage_percent'] > 50:
        print("\n   ⚠️  WARNING: Damage > 50% - Recovery unlikely")
        response = input("   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(0)

    # Step 3: Attempt recovery
    print(f"\n🔧 Step 3/4: Attempting recovery...")
    print(f"   Strategy: {damage_info['strategy']}")

    start_time = time.time()

    recovery = QRBruteForceRecovery(
        data_blocks=qr_data,
        version=args.version,
        ec_level=args.ec_level
    )

    if args.gpu:
        try:
            from qr_gpu_recovery import GPUBruteForceRecovery
            print("   ✓ Using GPU acceleration")
            recovery = GPUBruteForceRecovery(qr_data, args.version, args.ec_level)
        except ImportError:
            print("   ⚠️  GPU not available, using CPU")

    # Recovery with progress tracking
    if not args.no_progress:
        total_attempts = 2 ** min(args.max_flips * 8, 20)  # Cap for display
        pbar = tqdm(total=total_attempts, desc="   Testing combinations",
                   unit="combo", disable=False)
    else:
        pbar = None

    result = recovery.batch_test_combinations(
        base_bytes=qr_data,
        byte_positions=damage_info['critical_positions'],
        max_flips=args.max_flips,
        progress_callback=lambda: pbar.update(1) if pbar else None
    )

    if pbar:
        pbar.close()

    elapsed_time = time.time() - start_time

    # Step 4: Report results
    print(f"\n📊 Step 4/4: Results")
    print(f"   Time elapsed: {elapsed_time:.2f} seconds")

    if result:
        print("\n" + "🎉" * 35)
        print("✅ RECOVERY SUCCESSFUL!")
        print("🎉" * 35)
        print(f"\n   Recovered WIF Key: {result}")
        print(f"\n   ⚠️  SECURITY REMINDER:")
        print(f"   - Immediately transfer funds to new wallet")
        print(f"   - Never share this key")
        print(f"   - Delete this output securely")

        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(f"Recovered WIF Key: {result}\n")
                f.write(f"Recovery time: {elapsed_time:.2f}s\n")
                f.write(f"Image: {args.image}\n")
            print(f"\n   ✓ Saved to: {args.output}")
            print(f"   ⚠️  Remember to securely delete this file!")

    else:
        print("\n❌ RECOVERY FAILED")
        print("   Possible reasons:")
        print("   - Damage exceeds QR error correction capacity")
        print("   - Wrong QR version/error level specified")
        print("   - Image quality too poor")
        print("\n   Suggestions:")
        print("   - Try higher --max-flips value")
        print("   - Rescan with better quality")
        print("   - Use manual QR reconstruction (qrazybox.com)")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
