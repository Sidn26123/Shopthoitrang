package ptithcm.entity;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.ManyToOne;
import javax.persistence.Table;

@Entity
@Table(name="CTDONHANG")
public class CTDonHangEntity{
	@Id
	@GeneratedValue
	@Column(name = "MACTDH")
	private int maCTDH;
	@ManyToOne
	@JoinColumn(name = "MADH")
	private DonHangEntity donHang;
	@ManyToOne
	@JoinColumn(name = "MASP")
	private	SanPhamEntity sanPham;
	@Column(name = "DONGIA")
	private	int donGia;
	@Column(name="SOLUONG")
	private int soLuong;
	@Column(name = "SIZE") // Define the SIZE attribute
	private String size;
	@Column(name = "TRANGTHAIDANHGIA")
	
	private boolean trangThaiDanhGia;
	
	public boolean isTrangThaiDanhGia() {
		return trangThaiDanhGia;
	}
	public void setTrangThaiDanhGia(boolean trangThaiDanhGia) {
		this.trangThaiDanhGia = trangThaiDanhGia;
	}
	public int getMaCTDH() {
		return maCTDH;
	}
	public void setMaCTDH(int maCTDH) {
		this.maCTDH = maCTDH;
	}
	public DonHangEntity getDonHang() {
		return donHang;
	}
	public void setDonHang(DonHangEntity donHang) {
		this.donHang = donHang;
	}
	
	public int getDonGia() {
		return donGia;
	}
	public void setDonGia(int donGia) {
		this.donGia = donGia;
	}
	public int getSoLuong() {
		return soLuong;
	}
	public void setSoLuong(int soLuong) {
		this.soLuong = soLuong;
	}
	public SanPhamEntity getSanPham() {
		return sanPham;
	}
	public void setSanPham(SanPhamEntity sanPham) {
		this.sanPham = sanPham;
	}
	public String getMaSP() {
		return this.sanPham.getMaSP();
	}
	public String getSize() {
        return size;
    }

    public void setSize(String size) {
        this.size = size;
    }
}