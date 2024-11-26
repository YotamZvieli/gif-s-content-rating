import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Button,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Grid,
} from '@mui/material';
import { styled, keyframes } from '@mui/system';
import axios from 'axios';

// Styled Components
const Background = styled(Box)({
  minHeight: '100vh',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
});

const UploadCard = styled(Card)(({ theme }) => ({
  maxWidth: 500,
  margin: '0 auto',
  borderRadius: theme.shape.borderRadius,
  boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.1)',
  overflow: 'hidden',
}));

const UploadBox = styled(Box)(({ theme }) => ({
  border: '2px dashed rgba(255, 255, 255, 0.7)',
  borderRadius: '10px',
  padding: theme.spacing(4),
  textAlign: 'center',
  cursor: 'pointer',
  backgroundColor: 'rgba(255, 255, 255, 0.1)',
  transition: 'background-color 0.3s ease',
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
  },
}));

const Input = styled('input')({
  display: 'none',
});

const PreviewWrapper = styled(Box)({
  position: 'relative',
  textAlign: 'center',
  marginTop: '16px',
});

const PreviewImage = styled('img')({
  maxWidth: '100%',
  maxHeight: '200px',
  borderRadius: '8px',
});

const FadeIn = keyframes`
  from {
    opacity: 0;
    transform: scale(0.8);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
`;

const FeedbackIndicator = styled(Box)(({ type }) => ({
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  backgroundColor: type === 'success' ? 'rgba(0, 200, 0, 0.8)' : 'rgba(200, 0, 0, 0.8)',
  color: '#fff',
  borderRadius: '50%',
  width: '80px',
  height: '80px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  fontSize: '2rem',
  fontWeight: 'bold',
  animation: `${FadeIn} 0.5s ease-out`,
  zIndex: 1,
}));

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [rating, setRating] = useState(null);

  const handleFileChange = (event) => {
    setError(null);
    setRating(null);
    const file = event.target.files[0];
    if (file && file.type === 'image/gif') {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
    } else {
      setError('Please select a valid GIF file.');
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('No file selected.');
      return;
    }

    const formData = new FormData();
    formData.append('gif', selectedFile);

    try {
      setLoading(true);
      const response = await axios.post('https://shaggy-maps-pay.loca.lt/rate-gif', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'bypass-tunnel-reminder': '1',
        },
      });
      setRating(response.data.rating); // Expecting response with rating like "appropriate" or "inappropriate"
    } catch (err) {
      setError('Error uploading file.');
    } finally {
      setLoading(false);
    }
  };

  const renderFeedbackIndicator = () => {
    if (rating === 'Appropriate') {
      return <FeedbackIndicator type="success">✓</FeedbackIndicator>;
    } else if (rating === 'Inappropriate') {
      return <FeedbackIndicator type="error">✕</FeedbackIndicator>;
    }
    return null;
  };

  return (
    <Background>
      <UploadCard>
        <CardContent>
          <Typography variant="h4" align="center" gutterBottom color="white">
            GIF Classification
          </Typography>
          <Typography variant="subtitle1" align="center" color="white" paragraph>
            Upload a GIF to analyze its content using machine learning.
          </Typography>

          {error && <Alert severity="error">{error}</Alert>}

          <UploadBox>
            <label htmlFor="contained-button-file">
              <Input
                accept="image/gif"
                id="contained-button-file"
                type="file"
                onChange={handleFileChange}
              />
              <Button variant="contained" component="span" color="secondary">
                Choose GIF
              </Button>
            </label>
          </UploadBox>

          {preview && (
            <PreviewWrapper>
              {renderFeedbackIndicator()}
              <PreviewImage src={preview} alt="Selected GIF Preview" />
            </PreviewWrapper>
          )}

          <Grid container justifyContent="center" mt={2}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleUpload}
              disabled={loading || !selectedFile}
            >
              {loading ? <CircularProgress size={24} color="inherit" /> : 'Upload GIF'}
            </Button>
          </Grid>

          {rating && (
            <Alert
              severity={rating === 'Appropriate' ? 'success' : 'error'}
              style={{ marginTop: '16px' }}
            >
              The GIF is {rating === 'Appropriate' ? 'appropriate' : 'Inappropriate'}.
            </Alert>
          )}
        </CardContent>
      </UploadCard>
    </Background>
  );
}

export default App;
