import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Optimizations pour une app de lecture de BD/manga
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
      },
      {
        protocol: 'https',
        hostname: '**',
      },
    ],
    // Formats d'images optimisés
    formats: ['image/webp', 'image/avif'],
    // Tailles d'images communes pour les BD
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
  },

  // Support pour les fichiers statiques (modèles ONNX)
  webpack: (config, { isServer }) => {
    // Support pour les fichiers .onnx
    config.module.rules.push({
      test: /\.onnx$/,
      use: {
        loader: 'file-loader',
        options: {
          publicPath: '/_next/static/models/',
          outputPath: 'static/models/',
        },
      },
    });

    // Optimisations pour ONNX Runtime Web et PDF.js
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
        canvas: false,
      };
      
      // Exclure les modules problématiques du côté client
      config.externals = config.externals || [];
      config.externals.push({
        'canvas': 'canvas',
        'sharp': 'sharp',
      });
    }

    return config;
  },

  // Headers de sécurité et performance
  async headers() {
    return [
      {
        source: '/models/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
          {
            key: 'Cross-Origin-Embedder-Policy',
            value: 'require-corp',
          },
          {
            key: 'Cross-Origin-Opener-Policy',
            value: 'same-origin',
          },
        ],
      },
    ];
  },

  // Configuration PWA-ready
  experimental: {
    optimizePackageImports: ['lucide-react', 'framer-motion'],
  },

  // Configuration de build optimisée
  compress: true,
  poweredByHeader: false,
  
  // Support pour le mode standalone (Docker)
  output: 'standalone',
};

export default nextConfig;
