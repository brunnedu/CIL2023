from __future__ import annotations
from inspect import signature
import typing as t

class Injector:
    ''' 
        Basic Injector that distinguishes between

        Complex Data:
            - Arguments will be injected recursively (you register another injector that will be inovked
              whenever it's key is present in the data)
        
        Simple Data:
            - Arguments will be copied as is

        Returns the dictionary with all values injected recursively as much as possible.
    '''
    def __init__(self):
        self.injectors = {}

    def register(self, key : str, injector : Injector) -> Injector:
        ''' Registers a sub injector that will be invoked on the value for some key '''
        if key in self.injectors.keys():
            print(f'Injector {key} already exists. Will be overwritten!')

        self.injectors[key] = injector
        return self

    def build(self, data: t.Dict) -> object:
        ''' Parses a dictionary recursively '''

        # build all complex arguments
        arguments = {}
        for key,injector in self.injectors.items():
            if key in data.keys():
                injected = injector.build(data[key])
                if isinstance(injector, Proxy):
                    return injected
                else:
                    arguments[key] = injected
        
        # build all simple arguments
        for key,value in data.items():
            if key in self.injectors.keys():
                continue # is a complex argument
            
            arguments[key] = value

        return arguments

    
    def hierarchical_schema(self, depth : int = 0) -> str:
        ''' Returns a markdown schema of this injector '''
        md = ''
        depth_prefix = '&ensp;'*depth
        for key,injector in self.injectors.items():
            md += f'''<details><summary>{depth_prefix}{key}</summary>{depth_prefix}{injector.hierarchical_schema(depth + 1)}</details>\n'''
        
        return md
    
    # Some syntactic sugar
    def __setitem__(self, key : str, injector: Injector):
        self.register(key, injector)

    def __call__(self, data: t.Dict) -> object:
        return self.build(data)

class Proxy(Injector):
    ''' 
        Complex Injector that will construct an object of a specified type, using the recursively injected data as arguments.
        
        Analogously to the basic injector it distinguishes between

        Complex Data:
            - Arguments will be injected recursively (you register another injector that will be inovked
              whenever it's key is present in the data)
        
        Simple Data:
            - Arguments will be copied as is

        Returns the dictionary with all values injected recursively as much as possible.
    '''
    def __init__(self, ctor):
        super().__init__()

        self.ctor = ctor

    def build(self, data: t.Dict) -> object:
        ''' Parses a dictionary recursively '''
        arguments = super().build(data)
        return self.ctor(**arguments)
    
    def hierarchical_schema(self, depth : int = 0) -> str:
        ''' Returns a markdown schema of this injector '''

        args = signature(self.ctor).parameters
        if len(args.keys()) == 0: return ''

        depth_prefix = '&ensp;'*depth

        simple_args = ''.join([f'<br />{depth_prefix}&ensp;  - {p}' for v,p in signature(self.ctor).parameters.items() if v not in self.injectors.keys()])
        md = f'{depth_prefix}&ensp;Arguments: {simple_args} \n'
        for key,injector in self.injectors.items():
            md += f'''<details><summary>{depth_prefix}- {key}</summary>{depth_prefix}{injector.hierarchical_schema(depth + 1)}</details>\n'''

        return md
