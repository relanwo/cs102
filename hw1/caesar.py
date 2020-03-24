def key(keyword):
   
    if ("A" <= keyword <= "Z"):
        return ord(keyword) - 65
    else:
        return ord(keyword) - 97


def next_letter(letter):
    
    if ("A" <= letter <= "Z"):
        return chr(ord("A") + (ord(letter) - ord("A") + 3) % 26)
    if ("a" <= letter <= "z"):
        return chr(ord("a") + (ord(letter) - ord("a") + 3) % 26)
    else: 
        return letter

def encrypt_vigenere(plaintext, keyword):
    """
    >>> encrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> encrypt_vigenere("python", "a")
    'python'
    >>> encrypt_vigenere("ATTACKATDAWN", "LEMON")
    'LXFOPVEFRNHR'
    """
    # PUT YOUR CODE HERE
    ciphertext = ""
    k = 0

    for i in plaintext:
        ciphertext += next_letter(i, key(keyword[k % len(keyword)]))
        k += 1
    
    return ciphertext

def prev_letter(letter):
    
    if ("A" <= letter  <= "Z"):
        return chr(ord("A") + (ord(letter) - ord("A") - 3) % 26)
    if ("a" <= letter <= "z"):
        return chr(ord("a") + (ord(letter) - ord("a") - 3) % 26)
    else: 
        return letter

def decrypt_vigenere(ciphertext, keyword):
    """
    >>> decrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> decrypt_vigenere("python", "a")
    'python'
    >>> decrypt_vigenere("LXFOPVEFRNHR", "LEMON")
    'ATTACKATDAWN'
    """
    # PUT YOUR CODE HERE
    plaintext = ""
    k = 0

    for i in ciphertext:
        plaintext += prev_letter(i, key(keyword[k % len(keyword)]))
        k += 1

    return plaintext