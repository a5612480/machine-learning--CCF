a = "110"
p = []
for i in range(len(a)):
    p.append(a[len(a) - 1 - i])
# print(p)  //p = ['0','1','1']
# p是a字符串逆置后的结果
sum = 0
for i in range(len(p)):
    sum += int(p[i]) * pow(2,i)
print(sum) # 6

#进行十进制转换二进制
result = " "
while(sum != 0):
    ret = sum % 2
    sum = sum // 2
    result = str(ret) + result
print(result)

