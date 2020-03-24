def encrypt_vigenere(plaintext: str, keyword: str) -> str:

    """
        >>> encrypt_vigenere("PYTHON", "A")
        'PYTHON'
        >>> encrypt_vigenere("python", "a")
        'python'
        >>> encrypt_vigenere("ATTACKATDAWN", "LEMON")
        'LXFOPVEFRNHR'
        """

    keyword *= len(plaintext) // len(keyword) +1

    ciphertext = ''
    for i,j in enumerate(plaintext):

        if 'A' <= j <= 'Z' or 'a' <= j <= 'z':
            shift = ord(keyword[i % len(keyword)])
            if 'z' >= j >= 'a':
                shift -= ord('a')
            else:
                shift -= ord('A')
            code = ord(j) + shift

            if 'a' <= j <= 'z' and code > ord('z'):
                code -= 26
            elif 'A' <= j <= 'Z' and code > ord('Z'):
                code -= 26
            ciphertext += chr(code)
        else:
            ciphertext += j

    return ciphertext


print(encrypt_vigenere(input(), input()))


def decrypt_vigenere(ciphertext: str, keyword: str) -> str:

    """
       >>> decrypt_vigenere("PYTHON", "A")
       'PYTHON'
       >>> decrypt_vigenere("python", "a")
       'python'
       >>> decrypt_vigenere("LXFOPVEFRNHR", "LEMON")
       'ATTACKATDAWN'
       """

    plaintext = ""
    for i, j in enumerate(ciphertext):
        if 'A' <= j <= 'Z' or 'a' <= j <= 'z':
            shift = ord(keyword[i % len(keyword)])
            if 'z' >= j >= 'a':
                shift -= ord('a')
            else:
                shift -= ord('A')

            code = ord(j) - shift
            if code < ord('a') and 'a' <= j <= 'z':
                code += 26
            elif code < ord('A') and 'A' <= j <= 'Z':
                code += 26
            plaintext += chr(code)
        else:
            plaintext += j
    return plaintext


print(decrypt_vigenere(input(), input()))