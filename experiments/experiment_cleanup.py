import shelve

storage_path = "db/models_store.db"

with shelve.open(storage_path, writeback=True) as db:
    print("Aktualne klucze w bazie:", list(db.keys()))

    # Wyszukaj i usuń wszystkie modele ze statusem 'active'
    active_keys = [k for k in db.keys() if k.endswith("_active")]
    # Albo jeśli chcesz usunąć konkretnego użytkownika (np. 31):
    # active_keys = ["ppo_user_31_active"]

    for key in active_keys:
        del db[key]
        print(f"Usunięto klucz: {key}")

    print("Klucze po czyszczeniu:", list(db.keys()))