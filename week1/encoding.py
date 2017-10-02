## This is part one of this week's assignment. It looks at characters.
## Part two will involve tokenization of files into words.
## To turn in this assignment, you will create a zip file containing the `week1` directory.
## Some answers will involve adding code, and we will be able to see the output when we run the script. For others, you will need to add your response in comments.

### Compatibility check: Does this word look like greek? Does it print to the screen correctly?
greek = "μῆνιν"
print(greek)
print("--------------------------------------------------------")
# It looks like greek and prints correctly!

### 1. From characters to codepoints with `ord`.
###  This function takes a character and returns a numeric codepoint.
###  Python 3 uses Unicode natively.
print("Question 1")
print("The codepoint is {}".format(ord("A")))

### Modify the following two lines to print the codepoint in hexadecimal and zero-padded 8-bit binary.
### Useful: https://docs.python.org/3.6/library/string.html#format-specification-mini-language
print("The codepoint in hex is {:x}".format(ord("A"),'X'))
print("The codepoint in binary is {:08b}".format(ord("A")))
print("--------------------------------------------------------")

### 2. From codepoints to characters with chr().
print("Question 2")
print("The character is {}".format(chr(97)))
### copy the previous line to find the character for codepoint 210.
print("The character is {}".format(chr(210)))
print("--------------------------------------------------------")
### 3. From codepoints to bytes with encode()
### The following example string contains four codepoints.
english = "word"
### The encode() function maps the abstract codepoints to bytes, and returns
###  a byte array, which is not a string but may look like one.
print("Question 3")
print("English")
print(english.encode("utf-8"))
print(english.encode("ascii"))
print(english.encode("utf-16"))
print(english.encode("iso-8859-1"))
print('---------------------------------')
### a. Change the encoding to "ascii", "utf-16", and "iso-8859-1" (Latin 1). Describe what happens in the comment below.
### utf-8,"ascii", and "iso-8859-1" all return "b'word'". This is the byte version of the word as shown by the "b" prefice.
### utf-16 prints b'\xff\xfew\x00o\x00r\x00d\x00'. '\xff\xfe' is the byte order mark put by utf-16
### then since utf-16 uses 4 bytes instead of two it pads each character in w o r d with null bytes \00.
### http://unicodebook.readthedocs.io/unicode_encodings.html

### b. Try a more interesting example. What happens when you convert an Icelandic
###  string to ascii, utf-8, utf-16, and iso-8859-1? You may need to look carefully at the byte array.
icelandic = "orð"
print("Icelandic")
#print(icelandic.encode("ascii"))
print(icelandic.encode("utf-8"))
print(icelandic.encode("utf-16"))
print(icelandic.encode("iso-8859-1"))
print('---------------------------------')
### The interpreter throws an error becuase ð can't be represented in ascii
### UTF-8 returns b'or\xc3\xb0' where \xc3\xb0 is the codepoint for ð
### UTF-16 returns b'\xff\xfeo\x00r\x00\xf0\x00' where \xf0 is its codepoint for ð
### Latin-1's codepoint is the same as utf-16 but does not eed to pad with 0's. It
### returns b'or\xf0'


### c. Now use the `greek` variable from above, and try the same encodings.
print("Greek")
#print(greek.encode("ascii"))
print(greek.encode("utf-8"))
print(greek.encode("utf-16"))
#print(greek.encode("iso-8859-7"))
print("--------------------------------------------------------")
### Greek can't be encoded in ascii
### utf-8 prints b'\xce\xbc\xe1\xbf\x86\xce\xbd\xce\xb9\xce\xbd'
### utf-16 prints b'\xff\xfe\xbc\x03\xc6\x1f\xbd\x03\xb9\x03\xbd\x03'
### Latin-1 doesn't cover greek so it can't be encoded.
### Latin-7 or iso-8859-7 is specifically designed for modern greek
### but upon testing it, even it still does not cover all the characters
### http://www.sagehill.net/docbookxsl/CharEncoding.html

### 4. From bytes to codepoints with decode()
### To invert the transformation from bytes to codepoints we need to specify an encoding scheme. What happens if we get it wrong?
### a. Write code to encode the `greek` word in UTF-8, but then decode it as Latin 1, and print the result. What happens?
print("Question 4")
def decodeg(wrd):
    enc = wrd.encode("utf-8")
    dec = enc.decode("iso-8859-1")
    return dec

transf = decodeg(greek)
print("Transformation 1")
print(transf)
print("--------------------------------------------------------")
### The code first encodes acording to utf-8
### Howeever when trying to decode according to Latin-1 codepoints
### But Latin-1 does not support Greek so the result is
### "Î¼á¿Î½Î¹Î½"

### b. Do this transformation once again (encode as UTF-8, decode as Latin 1). What does it look like?
transf2 = decodeg(transf)
print("Transformation 2")
print(transf2)
print("--------------------------------------------------------")
### The result is even more garbled due to the fact that transf
### is Latin-1 while we are trying to encode in UTF-8. Then trying
### to decode again in Latin-1 gives "ÃÂ¼Ã¡Â¿ÂÃÂ½ÃÂ¹ÃÂ½"


### 5. Whitespace and punctuation. Consider this string:
stri = "This string has a tab\tand, as well, more\r\nthan one line."
### Write a for-loop that prints the codepoint for each character.
print("Question 5")
print("Codepoint for loop")
for i in stri:
    print(ord(i))

print("--------------------------------------------------------")


### 6. Something is strange about this string. Use the tools you've seen so far to inspect it. Is it what it appears?
weird_string = "Eﬃciency is the ﬁnal reﬂection"
### [description here, with any helpful code]
### The first thing I did was go character by character with my arrow keys
### discovering that "ﬃ","ﬁ", and "ﬂ" are just single characters.
print("Question 6")
print(ord("ﬃ"))
print(ord("ﬁ"))
print(ord("ﬂ"))
### Looking up the codepoints for these characters I found that they live in
### the unicode group "Alphabetic Presentation Forms"
### www.codetable.net
print("--------------------------------------------------------")

### 7. One of the responses I got Friday looked like this when I opened it.
text_with_Os = "He said ÒDonÕt worry!Ó, but I didnÕt believe him."
## In the comment below, describe what happened. What characters are not showing correctly, and why are they displaying as they are? Hint: figure out what OS the student was using.
### Punctuation characters are not showing correctly.
### " (left) shows as Ò
### " (right) shows as Ó
### ' shows as Õ
### The problem is that the student who wrote the respose was using a mac,
### or at least a computer using "Mac Roman". In Mac Roman "’" is D5, "”" is
### D3 and "“" is D2. However, on windows, or a computer using Latin-1, then
### D2 corresponds to "Ò", D3 to "Ó", and D5 to "Õ" which is why the characters
### were displayed incorrectly.

print("Question 7")
mystery = "Õ"
mystery2 = "Ò"
mystery3 = "Ó"
print(mystery.encode("utf-8"))
#print(mystery.encode("ascii"))
print(mystery.encode("utf-16"))
print(mystery.encode("iso-8859-1"))
print('---------------------------------')
windows1 = mystery.encode("iso-8859-1")
windows2 = mystery2.encode("iso-8859-1")
windows3 = mystery3.encode("iso-8859-1")
print(windows1.decode("mac-roman"))
print(windows2.decode("mac-roman"))
print(windows3.decode("mac-roman"))
print("--------------------------------------------------------")

### 8. The file "mystery.txt" is in a standard encoding, but which one?
###
###  a. Fill in the correct encoding in code below (it's not UTF8).
### UNCOMMENT THE FOLLOWING 3 LINES (it will cause a UnicodeDecodeError)
print("Problem 8")
with open("mystery.txt", encoding="iso-8859-5") as file:
   for line in file:
       print(line)
print("The text is in Cyrillic (Bulgarian)")
print("--------------------------------------------------------")
###  b. Figure out what language the text is in, and include it below. You may use any internet resource.
### The text is in Cyrillic Bulgarian. After taking random parts of the text an searching
### google for them, one of the most popular results was on bg.wikipedia.org,
### which is wikipedia in bulgarian.
