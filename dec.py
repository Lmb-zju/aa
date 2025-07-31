def fout(func):
    def fin(*args):
        print('1:', *args, '\n')
        return func(*args)
    return fin
@fout
def foo(a, b, c):
    print(a+b+c)

foo(1, 2, 3)
