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
} from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import Grid from "@mui/material/Grid";
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';

const LANGUAGES = [
  { value: "en", label: "English" },
  { value: "ar", label: "Arabic" },
  { value: "zh-cn", label: "Chinese (Simplified)" },
  { value: "zh-tw", label: "Chinese (Traditional)" },
  { value: "nl", label: "Dutch" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" },
  { value: "hi", label: "Hindi" },
  { value: "it", label: "Italian" },
  { value: "ja", label: "Japanese" },
  { value: "ko", label: "Korean" },
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
  const debounceTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [inputHeight, setInputHeight] = useState(60); // px, min height

  const wordCount = inputText.trim() ? inputText.trim().split(/\s+/).length : 0;
  const charCount = inputText.length;

  // Debounced auto-translate effect
  useEffect(() => {
    if (!inputText.trim()) {
      setOutputText("");
      return;
    }
    if (debounceTimeout.current) clearTimeout(debounceTimeout.current);
    debounceTimeout.current = setTimeout(async () => {
      try {
        const res = await fetch("https://cursor-first-test.onrender.com/translate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: inputText,
            source_lang: sourceLang,
            target_lang: targetLang,
          }),
        });
        if (!res.body) throw new Error("No response body");
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let result = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          result += decoder.decode(value, { stream: true });
          setOutputText(result);
        }
      } catch (err) {
        setOutputText("Translation failed.");
      }
    }, 600); // 600ms debounce
    return () => {
      if (debounceTimeout.current) clearTimeout(debounceTimeout.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputText, sourceLang, targetLang]);

  // Workaround: retrigger translation on window focus after inactivity
  useEffect(() => {
    const handleFocus = () => {
      if (inputText.trim()) {
        setInputText(prev => prev + " "); // trigger useEffect
        setTimeout(() => setInputText(prev => prev.trim()), 0); // restore original text
      }
    };
    window.addEventListener('focus', handleFocus);
    return () => window.removeEventListener('focus', handleFocus);
  }, [inputText]);

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
                disabled
                sx={{
                  fontWeight: 500,
                  fontSize: 16,
                  color: "#bbb",
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
                  <TextField
                    multiline
                    minRows={12}
                    maxRows={20}
                    fullWidth={true}
                    placeholder="Translation will appear here..."
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
                </Grid>
              </Grid>
            </Paper>
          )}
        </Container>
      </Box>
    </ThemeProvider>
  );
}
