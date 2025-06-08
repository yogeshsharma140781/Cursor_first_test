import { useState, useEffect, useRef } from "react";
import {
  Container,
  Box,
  Typography,
  Tabs,
  Tab,
  TextField,
  Button,
  MenuItem,
  Select,
  InputLabel,
  FormControl,
  Paper,
  CssBaseline,
} from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import Grid from "@mui/material/Grid";

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
  const [loading, setLoading] = useState(false);
  const debounceTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const wordCount = inputText.trim() ? inputText.trim().split(/\s+/).length : 0;
  const charCount = inputText.length;

  // Debounced auto-translate effect
  useEffect(() => {
    if (!inputText.trim()) {
      setOutputText("");
      setLoading(false);
      return;
    }
    setLoading(true);
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
      } finally {
        setLoading(false);
      }
    }, 600); // 600ms debounce
    return () => {
      if (debounceTimeout.current) clearTimeout(debounceTimeout.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputText, sourceLang, targetLang]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: "100vh", bgcolor: "#fff", display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Container maxWidth="sm" sx={{ py: 0, px: 0 }}>
          {/* Header */}
          <Box display="flex" alignItems="center" justifyContent="center" mb={4}>
            <img src="/Logo-full.svg" alt="Logo" style={{ height: 60, width: "auto", display: "block" }} />
          </Box>

          {/* Tabs */}
          <Box sx={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
            <Tabs
              value={tab}
              onChange={(_, v) => setTab(v)}
              variant="fullWidth"
              sx={{
                mb: 3,
                borderRadius: "24px",
                background: "#f5f6fa",
                minHeight: 44,
                width: 400,
                maxWidth: 400,
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
            <Paper elevation={0} sx={{ p: 3, bgcolor: "#fff" }}>
              <Grid container spacing={3}>
                <Grid size={12}>
                  <FormControl fullWidth>
                    <InputLabel id="source-lang-label">Source Language</InputLabel>
                    <Select
                      labelId="source-lang-label"
                      value={sourceLang}
                      label="Source Language"
                      onChange={e => setSourceLang(e.target.value)}
                    >
                      <MenuItem value="auto">Detect Language</MenuItem>
                      {LANGUAGES.map(lang => (
                        <MenuItem key={lang.value} value={lang.value}>
                          {lang.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid size={12}>
                  <TextField
                    multiline
                    minRows={6}
                    maxRows={10}
                    fullWidth
                    placeholder="Type or paste text here..."
                    value={inputText}
                    onChange={e => setInputText(e.target.value)}
                    variant="outlined"
                  />
                </Grid>
                <Grid size={12}>
                  <Typography variant="body2" color="#888" sx={{ mt: -1, mb: 0, fontSize: 13 }}>
                    {wordCount} words, {charCount} characters
                  </Typography>
                </Grid>
                <Grid size={12}>
                  <FormControl fullWidth>
                    <InputLabel id="target-lang-label">Target Language</InputLabel>
                    <Select
                      labelId="target-lang-label"
                      value={targetLang}
                      label="Target Language"
                      onChange={e => setTargetLang(e.target.value)}
                    >
                      {LANGUAGES.map(lang => (
                        <MenuItem key={lang.value} value={lang.value}>
                          {lang.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid size={12}>
                  <TextField
                    multiline
                    minRows={12}
                    maxRows={20}
                    fullWidth
                    placeholder="Translation will appear here..."
                    value={outputText}
                    InputProps={{
                      readOnly: true,
                    }}
                    variant="outlined"
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
