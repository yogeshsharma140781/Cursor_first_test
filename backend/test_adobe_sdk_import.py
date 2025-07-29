import importlib
try:
    import adobe
    print('Imported adobe from:', adobe.__file__)
    from adobe.pdfservices.operation.auth.credentials import Credentials
    print('✅ Adobe PDF Services SDK import successful')
except ImportError as e:
    print('❌ ImportError:', e) 