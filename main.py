import time
import sys
import os

# Importiamo i moduli dei vari step
# Nota: I file sono stati rinominati per essere importabili (niente numeri iniziali)
import step1_gen_data
import step3_train_lld
import step4_train_hld
import step5_evaluate

def run_pipeline():
    start_global = time.time()
    print("\n" + "="*60)
    print("   AVVIO QEC NEURAL DECODER PIPELINE (ONE SHOT)")
    print("="*60 + "\n")

    # --- FASE 1: DATI ---
    print("\n>>> [1/4] Esecuzione Generazione Dati (Stim)...")
    t0 = time.time()
    step1_gen_data.generate()
    print(f">>> Fase 1 completata in {time.time()-t0:.2f}s")

    # --- FASE 2: TRAINING LLD ---
    print("\n>>> [2/4] Esecuzione Training Low-Level Decoder (LLD)...")
    t0 = time.time()
    step3_train_lld.train_lld()
    print(f">>> Fase 2 completata in {time.time()-t0:.2f}s")

    # --- FASE 3: TRAINING HLD ---
    print("\n>>> [3/4] Esecuzione Training High-Level Decoder (HLD)...")
    t0 = time.time()
    step4_train_hld.train_hld()
    print(f">>> Fase 3 completata in {time.time()-t0:.2f}s")

    # --- FASE 4: VALUTAZIONE ---
    print("\n>>> [4/4] Valutazione Finale e Plotting...")
    t0 = time.time()
    step5_evaluate.evaluate()
    print(f">>> Fase 4 completata in {time.time()-t0:.2f}s")

    print("\n" + "="*60)
    print(f"   PIPELINE COMPLETATA CON SUCCESSO IN {time.time()-start_global:.2f}s")
    print(f"   I risultati sono in: {os.path.join(os.getcwd(), 'results')}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_pipeline()
