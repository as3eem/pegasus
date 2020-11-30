import tensorflow as tf
with tf.io.gfile.GFile("b.txt") as f:
    email_body = ""
    print(f)
    for line in f:
        print("sdf  qkndwlafdsbfvu-------", line)
        if line == "\n":
            break
        email_body += line
    
    print("----------------")
    print(email_body)
    print("----------------")
    line = next(f)
    subject = ""
    
    for line in f:
        print("----------------")
        print(line)
        print("----------------")
        if line == "\n":
            break
        subject += line
    print(subject)
