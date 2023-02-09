
def custom_decorator(func):
    def wrapper(*args, **kwargs):
        print("Something is happening before the function is called.")
        result = func(*args, **kwargs)
        print("Something is happening after the function is called.")
        return result
    return wrapper

@custom_decorator
def say_hello(name):
    print(f"Hello, {name}!")

say_hello("John")