try:
    import adobe.pdfservices.operation
    print('✅ pdfservices-sdk import SUCCESSFUL')
except ImportError as e:
    print('❌ pdfservices-sdk import FAILED:', e) 