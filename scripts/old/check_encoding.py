import os

def check_file_encoding(file_path):
    with open(file_path, "rb") as f:
        raw = f.read()
    try:
        raw.decode("utf-8")
        print(f"✅ {file_path} is UTF-8")
    except UnicodeDecodeError:
        print(f"❌ {file_path} is NOT UTF-8")

if __name__ == "__main__":
    data_dir = "data"
    for file in os.listdir(data_dir):
        if file.endswith(".md") or file.endswith(".txt"):
            check_file_encoding(os.path.join(data_dir, file))
