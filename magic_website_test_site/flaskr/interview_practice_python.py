class employee:
    
    def __init__(self,firstname,lastname,salary):
        self.firstname = firstname 
        self.lastname = lastname
        self.salary = salary
        self.email = f'{self.firstname}.{self.lastname}@kite.com'
        
    def giveRaise(self,salary):
        self.salary = salary
    
class developer(employee):
    
    def __init__(self, firstname, lastname, salary, programming_languages):
        super().__init__(firstname,lastname,salary)
        self.prog_langs = programming_languages
    
    def addLanguages(self,lang):
        self.prog_langs += [lang]
        
employee1 = employee("Jon","Smith",80000)

print(employee1.salary)

employee1.giveRaise(125422)

print(employee1.salary)

dev1 = developer("Joe","Montana", 421251, ["Python","C"])

print(dev1.salary)

print(dev1.prog_langs)

name = "Bob"
age = 25

string = "Hi my name is %s and I am %i years old!" % (name,age)
print(string)