import { useState, useEffect, useRef } from "react";
import {
  Container,
  Box,
  Typography,
  Tabs,
  Tab,
  TextField,
  MenuItem,
  Select,
  FormControl,
  Paper,
  CssBaseline,
  CircularProgress,
  Button,
  Alert,
  Switch,
  FormControlLabel,
} from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import Grid from "@mui/material/Grid";
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import ScannerIcon from '@mui/icons-material/Scanner';

const LANGUAGES = [
  { value: "en", label: "English" },
  { value: "ar", label: "Arabic" },
  { value: "nl", label: "Dutch" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" },
  { value: "hi", label: "Hindi" },
  { value: "it", label: "Italian" },
  { value: "pl", label: "Polish" },
  { value: "pt", label: "Portuguese" },
  { value: "ru", label: "Russian" },
  { value: "es", label: "Spanish" },
  { value: "tr", label: "Turkish" },
  { value: "uk", label: "Ukrainian" },
  { value: "vi", label: "Vietnamese" },
];

const theme = createTheme({
  typography: {
    fontFamily: "Inter, Arial, sans-serif",
    h4: {
      fontWeight: 700,
      color: "#1e90ff",
      letterSpacing: 0.5,
    },
  },
  palette: {
    primary: { main: "#1e90ff" },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 600,
          fontSize: 16,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          background: "#fafbfc",
          borderRadius: 8,
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          background: "#fff",
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 16,
        },
      },
    },
  },
});

export default function App() {
  const [tab, setTab] = useState(0);
  const [sourceLang, setSourceLang] = useState("auto");
  const [targetLang, setTargetLang] = useState("en");
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [isFocused, setIsFocused] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const debounceTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [inputHeight, setInputHeight] = useState(60); // px, min height
  const abortController = useRef<AbortController | null>(null);
  
  // PDF Translation state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isScannedPdf, setIsScannedPdf] = useState(false);
  const [isPdfTranslating, setIsPdfTranslating] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);
  const [pdfSuccess, setPdfSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const wordCount = inputText.trim() ? inputText.trim().split(/\s+/).length : 0;
  const charCount = inputText.length;

  // PDF handling functions
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      setSelectedFile(file);
      setPdfError(null);
      setPdfSuccess(null);
    } else {
      setPdfError('Please select a valid PDF file');
    }
  };

  const handlePdfTranslate = async () => {
    if (!selectedFile) {
      setPdfError('Please select a PDF file first');
      return;
    }

    setIsPdfTranslating(true);
    setPdfError(null);
    setPdfSuccess(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('source_lang', sourceLang);
      formData.append('target_lang', targetLang);

      const endpoint = isScannedPdf ? '/translate-scanned-pdf' : '/translate-pdf';
      const response = await fetch(`https://cursor-first-test.onrender.com${endpoint}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Translation failed: ${response.statusText}`);
      }

      // Handle file download
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `translated_${selectedFile.name}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      setPdfSuccess(`PDF translated successfully! File downloaded as translated_${selectedFile.name}`);
    } catch (error) {
      setPdfError(error instanceof Error ? error.message : 'Translation failed');
    } finally {
      setIsPdfTranslating(false);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setPdfError(null);
    setPdfSuccess(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Debounced auto-translate effect with better performance
  useEffect(() => {
    if (!inputText.trim()) {
      setOutputText("");
      setIsTranslating(false);
      return;
    }
    
    // Cancel previous request if still running
    if (abortController.current) {
      abortController.current.abort();
    }
    
    if (debounceTimeout.current) clearTimeout(debounceTimeout.current);
    
    debounceTimeout.current = setTimeout(async () => {
      setIsTranslating(true);
      abortController.current = new AbortController();
      
      try {
        const res = await fetch("https://cursor-first-test.onrender.com/translate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: inputText,
            source_lang: sourceLang,
            target_lang: targetLang,
          }),
          signal: abortController.current.signal,
        });
        
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        
        const result = await res.json();
        setOutputText(result.translation || "Translation failed.");
        
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          console.error('Translation error:', err);
          setOutputText("Translation failed. Please try again.");
        }
      } finally {
        setIsTranslating(false);
      }
    }, 1200); // Increased debounce time for better UX
    
    return () => {
      if (debounceTimeout.current) clearTimeout(debounceTimeout.current);
      if (abortController.current) abortController.current.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputText, sourceLang, targetLang]);

  // Shrink textarea height when user stops typing and text is short
  useEffect(() => {
    if (!inputText) {
      setInputHeight(60);
      return;
    }
    const timeout = setTimeout(() => {
      if (inputRef.current) {
        setInputHeight(Math.max(60, inputRef.current.scrollHeight));
      }
    }, 700);
    return () => clearTimeout(timeout);
  }, [inputText]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ bgcolor: "#fff", display: 'block', width: '100vw', minHeight: '100vh' }}>
        <Container maxWidth="sm" sx={{ py: 0, px: { xs: 2, sm: 0 }, display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%', maxWidth: 420, margin: '0 auto' }}>
          {/* Header */}
          <Box sx={{ width: '100%', maxWidth: 420, display: 'flex', justifyContent: 'center', alignItems: 'center', mt: 2, mb: 4, mx: 'auto' }}>
            <img src="/Logo-full.svg" alt="Logo" style={{ height: 60, width: "auto", display: "block" }} />
          </Box>

          {/* Tabs */}
          <Box sx={{ width: '100%', maxWidth: 420, display: 'flex', justifyContent: 'center', alignItems: 'center', mb: 3, mx: 'auto' }}>
            <Tabs
              value={tab}
              onChange={(_, v) => setTab(v)}
              variant="fullWidth"
              sx={{
                borderRadius: "24px",
                background: "#f5f6fa",
                minHeight: 44,
                width: '100%',
                maxWidth: 400,
                mx: 'auto',
                "& .MuiTabs-indicator": {
                  background: "#1e90ff",
                  borderRadius: 2,
                },
              }}
            >
              <Tab
                label="Text"
                sx={{
                  fontWeight: 500,
                  fontSize: 16,
                  color: tab === 0 ? "#1e90ff" : "#888",
                  minHeight: 44,
                  minWidth: 150,
                  textTransform: 'none',
                }}
              />
              <Tab
                label="Document"
                sx={{
                  fontWeight: 500,
                  fontSize: 16,
                  color: tab === 1 ? "#1e90ff" : "#888",
                  minHeight: 44,
                  minWidth: 150,
                  textTransform: 'none',
                }}
              />
            </Tabs>
          </Box>

          {tab === 0 && (
            <Paper elevation={0} sx={{ p: 3, bgcolor: "#fff", width: '100%', maxWidth: 420, mx: 'auto' }}>
              <Grid container spacing={3}>
                {/* Language selectors row */}
                <Grid size={12}>
                  <Box display="flex" alignItems="center" justifyContent="center" sx={{ width: '100%', maxWidth: 420, mx: 'auto' }}>
                    <FormControl sx={{ width: '48%', maxWidth: 192 }}>
                      <Select
                        labelId="source-lang-label"
                        value={sourceLang}
                        onChange={event => setSourceLang(event.target.value)}
                        sx={{ width: '100%', fontSize: 15, maxWidth: 192 }}
                        displayEmpty
                      >
                        <MenuItem value="auto">Detect Language</MenuItem>
                        {LANGUAGES.map(lang => (
                          <MenuItem key={lang.value} value={lang.value}>
                            {lang.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <Box display="flex" alignItems="center" justifyContent="center" sx={{ width: 20, mx: 1 }}>
                      <ArrowForwardIcon sx={{ fontSize: 20, color: '#222', display: 'block', mx: 'auto' }} />
                    </Box>
                    <FormControl sx={{ width: '48%', maxWidth: 192 }}>
                      <Select
                        labelId="target-lang-label"
                        value={targetLang}
                        onChange={event => setTargetLang(event.target.value)}
                        sx={{ width: '100%', fontSize: 15, maxWidth: 192 }}
                        displayEmpty
                      >
                        {LANGUAGES.map(lang => (
                          <MenuItem key={lang.value} value={lang.value}>
                            {lang.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Box>
                </Grid>
                <Grid size={12}>
                  <Paper
                    elevation={0}
                    sx={{
                      border: '1px solid #ddd',
                      borderRadius: 2,
                      p: 2.5,
                      bgcolor: '#fafafa',
                      boxSizing: 'border-box',
                      position: 'relative',
                      width: '100%',
                      maxWidth: 420,
                      mx: 'auto',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'flex-start',
                    }}
                  >
                    {/* Faux placeholder absolutely positioned at the top */}
                    {(!isFocused && inputText === "") && (
                      <span
                        style={{
                          position: 'absolute',
                          top: 32,
                          left: 20,
                          color: '#888',
                          fontStyle: 'italic',
                          fontSize: 20,
                          pointerEvents: 'none',
                          zIndex: 2,
                        }}
                      >
                        Type or paste text here...
                      </span>
                    )}
                    <textarea
                      ref={inputRef}
                      value={inputText}
                      onFocus={() => setIsFocused(true)}
                      onBlur={() => setIsFocused(false)}
                      onChange={event => {
                        setInputText(event.target.value);
                        const el = event.target;
                        setInputHeight(Math.max(60, el.scrollHeight));
                      }}
                      style={{
                        width: '100%',
                        height: inputHeight,
                        border: 'none',
                        outline: 'none',
                        resize: 'none',
                        background: 'transparent',
                        fontSize: 20,
                        marginTop: 16,
                        color: '#222',
                        fontFamily: 'inherit',
                        fontStyle: 'normal',
                        zIndex: 3,
                        position: 'relative',
                        textAlign: 'left',
                        transition: 'height 0.2s',
                      }}
                      autoFocus
                    />
                    {/* Word/character count left-aligned under input box */}
                    <Typography variant="body2" color="#888" sx={{ mt: 1, mb: 0, fontSize: 13, textAlign: 'left', width: '100%' }}>
                      {wordCount} words, {charCount} characters
                    </Typography>
                  </Paper>
                </Grid>
                <Grid size={12}>
                  <Box sx={{ position: 'relative' }}>
                    <TextField
                      multiline
                      minRows={12}
                      maxRows={20}
                      fullWidth={true}
                      placeholder=""
                      value={outputText}
                      InputProps={{
                        readOnly: true,
                        style: {
                          fontSize: 20,
                          color: '#007AFF',
                          fontWeight: 500,
                          lineHeight: 1.6,
                          width: '100%',
                          maxWidth: 420,
                          margin: '0 auto',
                          background: '#fafbfc',
                          borderRadius: 8,
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'flex-start',
                          textAlign: 'left',
                        },
                      }}
                      variant="outlined"
                      sx={{ width: '100%', maxWidth: 420, margin: '0 auto', background: '#fafbfc', borderRadius: 2, display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}
                    />
                    {/* Loading indicator */}
                    {isTranslating && (
                      <Box
                        sx={{
                          position: 'absolute',
                          top: '50%',
                          left: '50%',
                          transform: 'translate(-50%, -50%)',
                          display: 'flex',
                          alignItems: 'center',
                          gap: 1,
                        }}
                      >
                        <CircularProgress size={20} />
                        <Typography variant="body2" color="#888">
                          Translating...
                        </Typography>
                      </Box>
                    )}
                  </Box>
                </Grid>
              </Grid>
            </Paper>
          )}

          {tab === 1 && (
            <Paper elevation={0} sx={{ p: 3, bgcolor: "#fff", width: '100%', maxWidth: 420, mx: 'auto' }}>
              <Grid container spacing={3}>
                {/* Language selectors row */}
                <Grid size={12}>
                  <Box display="flex" alignItems="center" justifyContent="center" sx={{ width: '100%', maxWidth: 420, mx: 'auto' }}>
                    <FormControl sx={{ width: '48%', maxWidth: 192 }}>
                      <Select
                        labelId="source-lang-label"
                        value={sourceLang}
                        onChange={event => setSourceLang(event.target.value)}
                        sx={{ width: '100%', fontSize: 15, maxWidth: 192 }}
                        displayEmpty
                      >
                        <MenuItem value="auto">Detect Language</MenuItem>
                        {LANGUAGES.map(lang => (
                          <MenuItem key={lang.value} value={lang.value}>
                            {lang.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <Box display="flex" alignItems="center" justifyContent="center" sx={{ width: 20, mx: 1 }}>
                      <ArrowForwardIcon sx={{ fontSize: 20, color: '#222', display: 'block', mx: 'auto' }} />
                    </Box>
                    <FormControl sx={{ width: '48%', maxWidth: 192 }}>
                      <Select
                        labelId="target-lang-label"
                        value={targetLang}
                        onChange={event => setTargetLang(event.target.value)}
                        sx={{ width: '100%', fontSize: 15, maxWidth: 192 }}
                        displayEmpty
                      >
                        {LANGUAGES.map(lang => (
                          <MenuItem key={lang.value} value={lang.value}>
                            {lang.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Box>
                </Grid>

                {/* Scanned PDF Toggle */}
                <Grid size={12}>
                  <Box display="flex" justifyContent="center" sx={{ mb: 2 }}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={isScannedPdf}
                          onChange={(e) => setIsScannedPdf(e.target.checked)}
                          color="primary"
                        />
                      }
                      label={
                        <Box display="flex" alignItems="center" gap={1}>
                          <ScannerIcon sx={{ fontSize: 20 }} />
                          <Typography variant="body2">
                            Scanned PDF (OCR)
                          </Typography>
                        </Box>
                      }
                    />
                  </Box>
                  <Typography variant="caption" color="textSecondary" sx={{ display: 'block', textAlign: 'center', mb: 2 }}>
                    {isScannedPdf 
                      ? "Use OCR to extract text from scanned documents and images"
                      : "Translate PDFs with selectable text"
                    }
                  </Typography>
                </Grid>

                {/* File Upload */}
                <Grid size={12}>
                  <Paper
                    elevation={0}
                    sx={{
                      border: '2px dashed #ddd',
                      borderRadius: 2,
                      p: 4,
                      textAlign: 'center',
                      bgcolor: '#fafafa',
                      cursor: 'pointer',
                      '&:hover': {
                        borderColor: '#1e90ff',
                        bgcolor: '#f8f9ff',
                      },
                    }}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".pdf"
                      onChange={handleFileSelect}
                      style={{ display: 'none' }}
                    />
                    
                    <UploadFileIcon sx={{ fontSize: 48, color: '#666', mb: 2 }} />
                    
                    {selectedFile ? (
                      <Box>
                        <Typography variant="h6" color="primary" sx={{ mb: 1 }}>
                          {selectedFile.name}
                        </Typography>
                        <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                          {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                        </Typography>
                        <Button 
                          variant="outlined" 
                          size="small" 
                          onClick={(e) => {
                            e.stopPropagation();
                            clearFile();
                          }}
                        >
                          Change File
                        </Button>
                      </Box>
                    ) : (
                      <Box>
                        <Typography variant="h6" color="textPrimary" sx={{ mb: 1 }}>
                          Choose PDF File
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Click to browse or drag and drop your PDF here
                        </Typography>
                      </Box>
                    )}
                  </Paper>
                </Grid>

                {/* Error/Success Messages */}
                {pdfError && (
                  <Grid size={12}>
                    <Alert severity="error" onClose={() => setPdfError(null)}>
                      {pdfError}
                    </Alert>
                  </Grid>
                )}

                {pdfSuccess && (
                  <Grid size={12}>
                    <Alert severity="success" onClose={() => setPdfSuccess(null)}>
                      {pdfSuccess}
                    </Alert>
                  </Grid>
                )}

                {/* Translate Button */}
                <Grid size={12}>
                  <Button
                    variant="contained"
                    fullWidth
                    size="large"
                    onClick={handlePdfTranslate}
                    disabled={!selectedFile || isPdfTranslating}
                    sx={{
                      py: 1.5,
                      fontSize: 16,
                      fontWeight: 600,
                    }}
                  >
                    {isPdfTranslating ? (
                      <Box display="flex" alignItems="center" gap={1}>
                        <CircularProgress size={20} color="inherit" />
                        {isScannedPdf ? 'Processing with OCR...' : 'Translating...'}
                      </Box>
                    ) : (
                      `Translate ${isScannedPdf ? 'Scanned ' : ''}PDF`
                    )}
                  </Button>
                </Grid>
              </Grid>
            </Paper>
          )}
        </Container>
      </Box>
    </ThemeProvider>
  );
}
 