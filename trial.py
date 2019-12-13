from collections import deque, namedtuple, defaultdict

D = deque()
D.append(1)
D.append(2)
D.append(3)
D.append([5, 4])
controller = namedtuple('controller', ['istraining', 'queue'])
print(D.popleft())
print(controller)

c = controller(istraining=True, queue=D)
print(c)
c.queue.pop()
# c.istraining = False
print(c)

controller = defaultdict(lambda: None, istraining=True, queue=D)
print(controller)
print(controller['queue'])