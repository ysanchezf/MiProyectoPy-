d = {'c1':[4,1,{'c2':[ "es otro listado", { '0' :[3,1,['hola']]}]}]}
resultado = d['c1'][0] > d['c1'][2]['c2'][1]['0'][0]
print(resultado) 
