#!/usr/bin/env python3
"""
Master script to run all laboratory works
Student: Kazakova Victoria, Group 34-9
Course: Computer Modeling
"""

import sys
import subprocess
import os


def print_header():
    """Print header"""
    print("=" * 70)
    print("ЛАБОРАТОРНІ РОБОТИ 5-6: КОМП'ЮТЕРНЕ МОДЕЛЮВАННЯ")
    print("Студент: Казакова Вікторія, Група: 34-9")
    print("=" * 70)


def run_lab(lab_number, lab_dir, script_name):
    """Run a laboratory work"""
    print(f"\n{'=' * 70}")
    print(f"ЗАПУСК ЛАБОРАТОРНОЇ РОБОТИ {lab_number}")
    print("=" * 70)
    
    script_path = os.path.join(lab_dir, script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ Error: {script_path} not found!")
        return False
    
    try:
        # Change to lab directory and run script
        original_dir = os.getcwd()
        os.chdir(lab_dir)
        
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True
        )
        
        os.chdir(original_dir)
        
        if result.returncode == 0:
            print(f"\n✓ Лабораторна робота {lab_number} виконана успішно!")
            return True
        else:
            print(f"\n❌ Лабораторна робота {lab_number} завершилась з помилкою!")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running lab {lab_number}: {e}")
        os.chdir(original_dir)
        return False


def main():
    """Main function"""
    print_header()
    
    # Check if dependencies are installed
    try:
        import numpy
        import matplotlib
        import scipy
        print("\n✓ Всі залежності встановлені")
    except ImportError as e:
        print(f"\n❌ Відсутні залежності: {e}")
        print("\nВстановіть залежності командою:")
        print("  pip install -r requirements.txt")
        return 1
    
    # Menu
    print("\nОберіть, що запустити:")
    print("  1 - Лабораторна робота 5 (Молекулярна динаміка)")
    print("  2 - Лабораторна робота 6 (Монте-Карло)")
    print("  3 - Обидві лабораторні роботи")
    print("  q - Вихід")
    
    choice = input("\nВаш вибір: ").strip()
    
    success = True
    
    if choice == "1":
        success = run_lab(5, "lab5_molecular_dynamics", "molecular_dynamics.py")
    elif choice == "2":
        success = run_lab(6, "lab6_monte_carlo", "monte_carlo.py")
    elif choice == "3":
        success1 = run_lab(5, "lab5_molecular_dynamics", "molecular_dynamics.py")
        success2 = run_lab(6, "lab6_monte_carlo", "monte_carlo.py")
        success = success1 and success2
    elif choice.lower() == "q":
        print("\nДо побачення!")
        return 0
    else:
        print("\n❌ Невірний вибір!")
        return 1
    
    # Final summary
    print("\n" + "=" * 70)
    if success:
        print("✓ ВСІ СИМУЛЯЦІЇ ВИКОНАНІ УСПІШНО")
    else:
        print("❌ ДЕЯКІ СИМУЛЯЦІЇ ЗАВЕРШИЛИСЬ З ПОМИЛКАМИ")
    print("=" * 70)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
