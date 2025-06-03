import bcrypt

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


if __name__ == "__main__":
    username="Cubet"
    password="Cubet@123#"
    hashed = hash_password(password)

    print("\nAdd this to your secrets.toml file:")
    print(f"[passwords]\n{username} = \"{hashed}\"")